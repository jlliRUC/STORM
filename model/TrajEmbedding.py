import torch
import torch.nn as nn
import numpy as np
from SegEncoder import GAT, SegEmbedding
from copy import deepcopy as c


class TEmbedding(nn.Module):
    """
    ST2Vec.model_network.TimeEmbedding. Note that time_seqs here are already obtained time embeddings from Date2Vec.
    Thus, TimeEmbedding here is mainly for padding.
    """
    def __init__(self, date2vec_size):
        super(TEmbedding, self).__init__()
        self.date2vec_size = date2vec_size

    def forward(self, time_seqs, traj_lengths):
        """
        padding and timestamp series embedding
        :param time_seqs: list [batch, timestamp_seq]
        :return: packed_input
        """
        batch_size = len(time_seqs)
        max_length = max(traj_lengths)

        vec_time_seqs = []
        for time_one in time_seqs:
            # time_one: (seq_len, self.date2vec_size)
            if len(time_one) < max_length:
                time_one = np.concatenate((time_one, np.array([[0 for i in range(self.date2vec_size)]] * (max_length-len(time_one)))), axis=0)
            vec_time_seqs.append(time_one)
        vec_time_seqs = np.array(vec_time_seqs)

        # prepare sequence tensor
        embedded_seq_tensor = torch.zeros((batch_size, max(traj_lengths), self.date2vec_size), dtype=torch.float)
        # time_seqs = torch.tensor(time_seqs).to(self.device)
        vec_time_seqs = torch.tensor(vec_time_seqs, dtype=torch.float64).cuda()

        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(vec_time_seqs, traj_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seq[:seqlen]

        # move to cuda device
        embedded_seq_tensor = embedded_seq_tensor.cuda()

        return embedded_seq_tensor


class SEmbedding(nn.Module):
    def __init__(self, configs, load_segment_embedding=False, segment_finetune=True, edge_dim=None):
        """
        Case 1: We train the segment embedding from scratch
        (1) rs: [segment features, edge_index, edge_weights]
        (2) load_segment_embedding: False
        (3) segment_finetune: True
        (4) self.gcn.input_size: sum of the seg_{feature}_dim
        Case 2: We finetune the existing segment embedding (e.g. node2vec as in ST2Vec, or SARN)
        (1) rs: [segment embedding, edge_index, edge_weights]
        (2) load_segment_embedding: True
        (3) segment_finetune: True
        (4) self.gcn.input_size: seg_size
        Case 3: We use the existing segment embedding directly
        (1) rs: [segment embedding, edge_index, edge_weights]
        (2) load_segment_embedding: True
        (3) segment_finetune: False
        (4) We don't have self.gcn
        """

        super(SEmbedding, self).__init__()

        self.embed_size = configs.seg_size
        self.load_segment_embedding = load_segment_embedding
        self.segment_finetune = segment_finetune
        self.edge_dim = edge_dim
        self.device = configs.device

        if load_segment_embedding:
            if segment_finetune:
                self.gcn = GAT(
                    input_size=self.embed_size,
                    hidden_size=self.embed_size,
                    output_size=self.embed_size,
                    num_heads=configs.num_heads,
                    dropput=configs.dropout,
                    num_layers=configs.num_gat_layer,
                    edge_dim=edge_dim)
            else:
                print("Warning: We are now using the pretrained segment embedding directly")
        else:
            self.seg_embed = SegEmbedding(configs)
            self.gcn = GAT(input_size=configs.seg_cls_dim + configs.seg_length_dim + configs.seg_radian_dim + configs.seg_loc_dim * 2,
                           hidden_size=self.embed_size,
                           output_size=self.embed_size,
                           num_heads=configs.num_heads,
                           dropput=configs.dropout,
                           num_layers=configs.num_gat_layer,
                           edge_dim=edge_dim)

    def forward(self, rs, rs_ids_list_ori, traj_lengths):
        """
        graph: [segment_id_list, segment_features_list, edge_index, edge_list]
        """
        rs_ids_list = c(rs_ids_list_ori)
        if not isinstance(rs_ids_list, list):
            rs_ids_list = list(rs_ids_list)

        # segment embedding
        if self.load_segment_embedding:
            if self.segment_finetune:
                if self.edge_dim is not None:
                    seg_embeddings = self.gcn(rs[0], rs[1], rs[2])
                else:
                    seg_embeddings = self.gcn(rs[0], rs[1])
            else:
                seg_embeddings = rs[0]
        else:
            seg_embeddings = self.seg_embed(rs[0])  # [num_segs, feature_dim]
            seg_embeddings = self.gcn(seg_embeddings, rs[1], rs[2])

        # seg_embeddings: [num_segs, seg_size]

        batch_size = len(rs_ids_list)

        for traj_one in rs_ids_list:
            traj_one += [0] * (max(traj_lengths) - len(traj_one))
        embedded_seq_tensor = torch.zeros((batch_size, max(traj_lengths), self.embed_size), dtype=torch.float32)

        traj_seqs = torch.tensor(rs_ids_list).to(self.device)
        del rs_ids_list
        # get embedding for trajectory embeddings
        for idx, (seq, seqlen) in enumerate(zip(traj_seqs, traj_lengths)):
            embedded_seq_tensor[idx, :seqlen] = seg_embeddings.index_select(0, seq[:seqlen])

        # move to cuda device
        embedded_seq_tensor = embedded_seq_tensor.to(self.device)

        return embedded_seq_tensor

