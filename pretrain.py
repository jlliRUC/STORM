import logging
import inspect
import time
from copy import deepcopy as c
import shutil

from data_utils import *

from config import Config
from model.ContrastiveLearning import ProjectionHead, CL
from model.TrajEncoder import *
from model.TrajEmbedding import SEmbedding, TEmbedding
from model.utils_model import *
from losses import get_loss


class Storm:
    def __init__(self, configs):
        self.configs = configs
        attn = MultiHeadedAttention(configs.num_heads, configs.embed_size)
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
        self.model = CL(encoder, projector)

    def load_dataset(self):
        self.train_datasets = CLDataLoader(dataset_name=self.configs.dataset_name,
                                           num_train=self.configs.num_train,
                                           num_val=self.configs.num_val,
                                           augmentation_list=["aug", "trim_rate_random",
                                                              "temporal_distortion_rate_random"],
                                           istrain=True,
                                           batch_size=self.configs.batch_size,
                                           shuffle=True)
        logging.info("Loading train datasets...")
        self.train_datasets.load()
        logging.info(f"Train dataset size: {self.train_datasets.num_traj}")

        self.valid_datasets = CLDataLoader(dataset_name=self.configs.dataset_name,
                                           num_train=self.configs.num_train,
                                           num_val=self.configs.num_val,
                                           augmentation_list=["aug", "trim_rate_random",
                                                              "temporal_distortion_rate_random"],
                                           istrain=False,
                                           batch_size=self.configs.batch_size,
                                           shuffle=True)
        logging.info("Loading valid data...")
        self.valid_datasets.load()
        logging.info(f"Valid dataset size: {self.valid_datasets.num_traj}")

        # Load road network info
        self.rs = get_segment_embedding(configs)  # (rs_embeddings/features, edge_index, edge_weights)
        logging.info("Loaded road network info...")

    def train(self):
        # Initialize logging
        # Load train and val dataset
        self.load_dataset()

        # Load model to device
        if self.configs.cuda and torch.cuda.is_available():
            logging.info("=> Training with GPU")
            self.model.cuda()
        else:
            logging.info("=> Training with CPU")
        logging.info(f"model {self.model}")
        print(f"model #parameters")
        get_parameter_number(self.model)

        # optimizer
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=configs.learning_rate,
                                     weight_decay=configs.learning_rate_decay)

        # load model state and optmizer state
        if os.path.isfile(self.configs.checkpoint):
            logging.info(f"=> Loading checkpoint {self.configs.checkpoint}")
            checkpoint = torch.load(configs.checkpoint)
            start_epoch = checkpoint["epoch"]
            start_iteration = checkpoint["iteration"] + 1
            best_valid_loss = checkpoint["best_valid_loss"]
            status = checkpoint["status"]  # The best loss has been stayed for n_status iterations
            self.model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            logging.info(f"=> No checkpoint found at {self.configs.checkpoint}")
            best_valid_loss = float('inf')
            init_parameters(self.model)
            print("=> Initialized the parameters...")
            start_epoch = 0
            start_iteration = 0
            status = 0

        # Training records initialization
        torch.autograd.set_detect_anomaly(True)
        start = time.time()
        epoch_start = time.time()
        train_gpu_usage = []
        train_ram_usage = []
        logging.info("=> Begin training")

        for current_epoch in range(start_epoch, self.configs.epochs):
            adjust_learning_rate(optimizer, self.configs.learning_rate, current_epoch, self.configs.epochs)
            if current_epoch != start_epoch:
                self.train_datasets.shuffle = True
                start_iteration = 0
                status = 0
            num_iteration = self.train_datasets.num_traj // self.configs.batch_size - start_iteration
            logging.info(f"[Training] Epoch {current_epoch} is training.")
            logging.info(f"[Training] {num_iteration} iterations will be done")
            self.train_datasets.start = start_iteration * configs.batch_size
            while True:
                optimizer.zero_grad()
                # batch dataset
                s_list, t_list, st_list, lengths_list = self.train_datasets.get_one_batch()
                if s_list is None:
                    logging.info(f"[Training] Finish epoch {current_epoch}.")
                    break

                # modeling
                output_list = [self.model(s, t, self.rs, lengths, st) for s, t, st, lengths in
                               zip(s_list, t_list, st_list, lengths_list)]
                h_list, z_list = [output[0] for output in output_list], [output[1] for output in output_list]
                features = torch.cat(z_list, dim=0).to(configs.device)

                # InfoNCE Loss
                criteration = get_loss(self.configs.loss)
                loss = criteration(self.configs.batch_size,
                                   self.configs.n_views,
                                   self.configs.temperature,
                                   features,
                                   self.configs.device)
                loss.backward()
                optimizer.step()
                start_iteration += 1

            train_gpu_usage.append(GPUInfo.mem()[0])
            train_ram_usage.append(RAMInfo.mem())

            # print
            logging.info(f"[Training] Epoch {current_epoch}: Contrastive loss {loss:3f}")
            logging.info(
                f"[Training] Epoch {current_epoch} costs: {time.time() - epoch_start:.4f} s, each iteration costs {(time.time() - epoch_start) / num_iteration:.4f} s")
            epoch_start = time.time()

            # valid
            valid_loss = self.validate()
            logging.info(f"[Training] Epoch {current_epoch}: validation loss {valid_loss}")
            logging.info(
                f"[Training] Epoch {current_epoch}: GPU usage {GPUInfo.mem()[0]} MB, RAM usage {RAMInfo.mem()} MB")
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                logging.info(f"[Training] Best model with valid loss {best_valid_loss} at epoch {current_epoch}")
                is_best = True
                status = 0
            else:
                is_best = False
                status += 1
                if status >= configs.best_threshold:
                    logging.info(f"[Training] No improvement after {self.configs.best_threshold} epochs, training stops")
                    break
            self.save_checkpoint({"epoch": current_epoch,
                                  "iteration": start_iteration,
                                  "best_valid_loss": best_valid_loss,
                                  "status": status,
                                  "model": self.model.state_dict(),
                                  "optimizer": optimizer.state_dict()
                                  }, is_best)

        logging.info(f"[Training] Training costs: {time.time() - start:.4f}s")
        logging.info(f"[Training] Training avg GPU usage {np.array(train_gpu_usage).mean()} MB, avg RAM usage {np.array(train_ram_usage).mean()} MB")

    def validate(self):
        self.model.eval()
        self.valid_datasets.start = 0
        num_iteration = 0

        total_loss = 0
        while True:
            s_list, t_list, st_list, lengths_list = self.valid_datasets.get_one_batch()
            if s_list is None:
                break
            with torch.no_grad():
                num_iteration += 1
                output_list = [self.model(s, t, self.rs, lengths, st) for s, t, st, lengths in
                               zip(s_list, t_list, st_list, lengths_list)]

                h_list, z_list = [output[0] for output in output_list], [output[1] for output in output_list]
                features = torch.cat(z_list, dim=0).to(configs.device)
                criteration = get_loss(self.configs.loss)
                total_loss += criteration(configs.batch_size, configs.n_views, configs.temperature, features,
                                          configs.device)
        # switch back to training mode
        self.model.train()
        return total_loss / num_iteration

    def save_checkpoint(self, state, is_best):
        torch.save(state, self.configs.checkpoint)
        if is_best:
            shutil.copyfile(self.configs.checkpoint, self.configs.bestpoint)

    def encode(self, vec_file):
        if os.path.isfile(self.configs.bestpoint):
            print("=> loading checkpoint '{}'".format(self.configs.bestpoint))
            checkpoint = torch.load(self.configs.bestpoint)
            self.model.load_state_dict(checkpoint["model"])
        else:
            print("=> no checkpoint found at '{}'".format(self.configs.bestpoint))
            return 0
        print(self.model)
        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

        # Load exp dataset
        vecs = []
        scaner = EvalDataLoader(self.configs.dataset_name, self.configs.batch_size)
        scaner.load()
        i = 0
        while True:
            if i % 100 == 0:
                print("Batch {}: Encoding {} road segments...".format(i, configs.batch_size * i))
            i = i + 1
            # After embedding, only input the original node list into loss function (They're full-covered)
            s, t, st, lengths = scaner.get_one_batch()
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
    task = Storm(configs)
    task.train()
