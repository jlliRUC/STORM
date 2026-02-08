import os
import time
from copy import deepcopy as c
import shutil


from config import Config
from data_utils import FinetuneDataLoader
from dataset_preparation import DataLoader
from model.ContrastiveLearning import ProjectionHead, CL
from model.TrajEncoder import *
from model.TrajEmbedding import SEmbedding, TEmbedding
from model.utils_model import *
from model.FinetuneModel import SimilarityLearning
from pretrain import get_segment_embedding
from losses import get_SLloss
import test_method


def save_checkpoint(state, is_best, configs):
    torch.save(state, configs.similarity_learning_checkpoint)
    if is_best:
        shutil.copyfile(configs.similarity_learning_checkpoint, configs.similarity_learning_bestpoint)


class Finetune:
    def __init__(self, configs):
        self.configs = configs
        attn = MultiHeadedAttention(configs.num_heads, configs.embed_size, extra_weights=True, weights_dim=configs.embed_size)
        #attn = MultiHeadedAttention(configs.num_heads, configs.embed_size)
        ff = PositionwiseFeedForward(configs.embed_size, 2048, configs.dropout)
        encoder = STEncoder(embed_S=SEmbedding(configs,
                                               load_segment_embedding=configs.load_segment_embedding,
                                               segment_finetune=configs.segment_finetune,
                                               edge_dim=configs.edge_dim),
                            embed_T=TEmbedding(configs.date2vec_size),
                            embed_P=PositionalEncoding(configs.embed_size, configs.dropout),
                            layer=EncoderLayer(configs.embed_size, c(attn), c(ff), configs.dropout),
                            embedding_size=configs.embed_size,
                            hidden_size=configs.hidden_size,
                            output_size=configs.output_size,
                            num_layers=configs.num_layers,
                            dropout=configs.dropout)
        projector = ProjectionHead(configs.embed_size, int(configs.embed_size / 2), int(configs.embed_size / 4),
                                   batch_norm=True)
        pretrained_model = CL(encoder, projector)

        if self.configs.task == "similarity learning":
            self.model = SimilarityLearning(pretrained_model, pretrained_model.encoder.hidden_size)
            self.checkpoint = self.configs.similarity_learning_checkpoint
            self.bestpoint = self.configs.similarity_learning_bestpoint
        else:
            print("Unknown finetune task")

    def load_dataset(self):
        self.finetune_datasets = FinetuneDataLoader(self.configs.dataset_name, self.configs.finetune_batch_size, self.configs.num_finetune, self.configs.distance_type, False)
        self.finetune_datasets.load(self.configs)
        print(f"Finetune dataset size: {self.finetune_datasets.num_traj}...")
        self.rs = get_segment_embedding(self.configs)

    def train(self):
        # Load dataset
        self.load_dataset()

        # Load pretrain model
        if self.configs.load_pretrain:
            if os.path.isfile(self.configs.bestpoint):
                print("=> loading checkpoint '{}'".format(self.configs.bestpoint))
                checkpoint = torch.load(self.configs.bestpoint)
                self.model.pretrained_model.load_state_dict(checkpoint["model"])
            else:
                print("=> no checkpoint found at '{}'".format(self.configs.bestpoint))
                return 0
            if not self.configs.finetune:
                for param in self.model.pretrained_model.parameters():
                    param.requires_grad = False

        # Load model to device
        if self.configs.cuda and torch.cuda.is_available():
            print("=> Training with GPU")
            self.model.cuda()
        else:
            print("=> Training with CPU")
        print(f"model {self.model}")
        print(f"model #parameters")
        get_parameter_number(self.model)

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.configs.finetune_learning_rate,
                                     weight_decay=self.configs.finetune_learning_rate_decay)

        # load model state and optmizer state
        if not self.configs.retrain and os.path.isfile(self.checkpoint):
            print("=> loading checkpoint '{}'".format(self.checkpoint))
            checkpoint = torch.load(self.checkpoint)
            start_epoch = checkpoint["epoch"]
            start_iteration = 0
            best_train_loss = checkpoint["best_train_loss"]
            status = checkpoint["status"]  # The best loss has been stayed for $status iterations
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            print("=> no checkpoint found at '{}'".format(self.checkpoint))
            best_train_loss = float('inf')
            init_parameters(self.model)
            print("=> initialized the parameters...")
            start_epoch = 0
            start_iteration = 0
            status = 0

        # Training records initialization
        #torch.autograd.set_detect_anomaly(True)
        start = time.time()
        train_gpu_usage = []
        train_ram_usage = []
        print("=> Begin training")

        for current_epoch in range(start_epoch, self.configs.finetune_epochs):
            epoch_start = time.time()
            if current_epoch != start_epoch:
                start_iteration = 0
                status = 0
            num_iteration = self.finetune_datasets.num_traj // self.configs.finetune_batch_size - start_iteration
            print(f"[Training] Epoch {current_epoch} is training.")
            print(f"[Training] {num_iteration} iterations will be done")
            self.finetune_datasets.start = start_iteration * self.configs.finetune_batch_size
            while True:
                optimizer.zero_grad()

                # batch dataset
                s_list, t_list, st_list, lengths_list, ground_truth = self.finetune_datasets.get_one_batch()

                if s_list is None:
                    print(f"[Training] Finish epoch {current_epoch}.")
                    break

                # modeling
                embeddings_list = [self.model(s_list[temp_idx], t_list[temp_idx], self.rs, lengths_list[temp_idx], st_list[temp_idx]) for temp_idx in range(len(s_list))]

                criterion = get_SLloss(self.configs.SLloss)
                ground_truth = ground_truth.to(self.configs.device)

                loss = criterion(embeddings_list, ground_truth)
                loss.backward()
                optimizer.step()
                start_iteration += 1

            train_gpu_usage.append(GPUInfo.mem()[0])
            train_ram_usage.append(RAMInfo.mem())

            # print
            print(
                f"[Training] Epoch {current_epoch} costs: {time.time() - epoch_start:.4f}, each iteration costs {(time.time() - epoch_start) / num_iteration:.4f} s")
            print(f"[Training] Epoch {current_epoch}: training loss {loss:.4f}")
            # valid
            valid_loss = self.validate()
            print(
                f"[Training] Epoch {current_epoch}: H@10 {np.round(valid_loss[0], 4)}, H@50 {np.round(valid_loss[1], 4)}, R10@50 {np.round(valid_loss[2], 4)}")
            print(
                f"[Training] Epoch {current_epoch}: GPU usage {GPUInfo.mem()[0]} MB, RAM usage {RAMInfo.mem()} MB")
            if loss.item() < best_train_loss:
                best_train_loss = loss.item()
                print(f"[Training] Best model with training loss {best_train_loss} at epoch {current_epoch}")
                is_best = True
                status = 0
            else:
                is_best = False
                status += 1
                if status >= self.configs.best_threshold:
                    print(f"[Training] No improvement after {self.configs.best_threshold} epochs, training stops")
                    break
            self.save_checkpoint({"epoch": current_epoch,
                                  "iteration": start_iteration,
                                  "best_train_loss": best_train_loss,
                                  "status": status,
                                  "model": self.model.state_dict(),
                                  "optimizer": optimizer.state_dict()
                                  }, is_best)
        print(f"[Training] Training costs: {time.time() - start:.4f}s")
        print(
            f"[Training] Training avg GPU usage {np.array(train_gpu_usage).mean()} MB, avg RAM usage {np.array(train_ram_usage).mean()} MB")

        # test
        test_loss = self.test()
        print(
            f"[Training] Test: H@10 {np.round(test_loss[0], 4)}, H@50 {np.round(test_loss[1], 4)}, R10@50 {np.round(test_loss[2], 4)}")

    def validate(self):
        self.model.eval()
        with torch.no_grad():
            vali_node_list, vali_rs_list, vali_time_list, vali_st_list, vali_d2vec_list = DataLoader(self.configs).load(
                load_part='vali')
            vali_lengths = torch.LongTensor(list(map(len, vali_rs_list)))
            print(f'We have {len(vali_node_list)} trajectories')
            t1 = time.time()
            embedding_vali = test_method.compute_embedding(net=self.model,
                                                           test_traj=vali_rs_list,
                                                           test_time=vali_d2vec_list,
                                                           road_network=self.rs,
                                                           test_length=vali_lengths,
                                                           test_st=vali_st_list,
                                                           test_batch=self.configs.finetune_batch_size)
            t2 = time.time()
            print(f"Inference time: {t2 - t1} s")
            acc = test_method.test_model(self.configs, embedding_vali, isvali=True)
            print(f"similarity computation time: {time.time() - t2} s")
        self.model.train()

        return acc

    def test(self):
        self.model.eval()
        with torch.no_grad():
            test_node_list, test_rs_list, test_time_list, test_st_list, test_d2vec_list = DataLoader(self.configs).load(
                load_part='test')
            test_lengths = torch.LongTensor(list(map(len, test_rs_list)))
            embedding_test = test_method.compute_embedding(net=self.model,
                                                           test_traj=test_rs_list,
                                                           test_time=test_d2vec_list,
                                                           road_network=self.rs,
                                                           test_length=test_lengths,
                                                           test_st=test_st_list,
                                                           test_batch=self.configs.finetune_batch_size)
            acc = test_method.test_model(self.configs, embedding_test, isvali=False)
        self.model.train()

        return acc

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.checkpoint)
        if is_best:
            shutil.copyfile(self.checkpoint, self.bestpoint)

    def encode(self, vec_file):
        if os.path.isfile(self.bestpoint):
            print("=> loading checkpoint '{}'".format(self.bestpoint))
            checkpoint = torch.load(self.bestpoint)
            self.model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(self.bestpoint))
            return 0
        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        # Load exp dataset
        vecs = []
        scaner = FinetuneDataLoader(self.configs.dataset_name, self.configs.batch_size, self.configs.num_test, "TP", False)
        scaner.load()
        i = 0
        while True:
            if i % 100 == 0:
                print("Batch {}: Encoding {} road segments...".format(i, configs.batch_size * i))
            i = i + 1
            s, t, st, sub_ground_truth, lengths = scaner.get_one_batch()
            if s is None: break
            h = self.model(s, t, self.rs, lengths, st)
            vecs.append(h.cpu().data)

        vecs = torch.cat(vecs).contiguous()  # # (num_seqs, hidden_size)
        path = vec_file
        print(f"=> saving vectors {vecs.shape} into {path}")
        with open(path, "wb") as f:
            pickle.dump(vecs, f)


if __name__ == "__main__":
    configs = Config()
    configs.dataset_update({"dataset_name": "porto"})
    configs.config_device()
    configs.config_train()
    task = Finetune(configs)
    task.train()


