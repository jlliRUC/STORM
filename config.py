import os
import torch
import pickle


class Config:
    def __init__(self):
        # Important settings to modify
        # # Main exp
        self.retrain = True
        self.dataset_name = "porto"
        self.distance_type = "TP"
        # # Ablation study
        self.load_segment_embedding = False
        self.segment_embedding_type = "node2vec"  # If we have pretrained segment embedding
        self.segment_finetune = True
        self.edge_dim = 1  # If consider the edge weight, then it's 1, otherwise it's None
        self.loss = "pos-out-single"  # CL loss
        self.n_views = 3  # CL
        self.num_companion = 5  # Each sample has num_companion pos and neg respectively
        self.SLloss = "difficulty"
        # # Parameter study
        self.epochs = 10  # CL
        self.finetune_epochs = 50
        self.finetune_batch_size = 32
        # change the training size in each dataset
        self.hidden_size = 128
        self.embed_size = 128
        self.output_size = 128
        self.num_layers = 1

        # Default settings
        self.root_dir = "/home/jiali/STORM"
        self.data_path = os.path.join(self.root_dir, f"data/")

        # Preprocessing
        self.min_length = 30  # Length filter for trajectory data
        self.max_length = 2000  # transformers need a small max_length to be fast

        # Road network feature embedding
        self.seg_cls_dim = 16
        self.seg_length_dim = 16
        self.seg_radian_dim = 16
        self.seg_loc_dim = 32
        self.seg_size = 64  # 128 for SARN
        self.num_gat_layer = 1
        self.cell_size = 100
        self.seg_length_unit = 5
        self.seg_radian_unit = 0.174533  # 10 degree
        self.seg_diss_unit = 5
        self.seg_tis_unit = 10
        self.seg_speed_unit = 5
        self.seg_turns_unit = 5
        self.seg_aspeed_unit = 2

        # Time Embedding
        self.date2vec_size = 64  # This should be fixed to 64 following Time2Vec

        # Trajectory modeling
        self.model_name = "STORM"
        self.model_settings = "main_evaluation"
        self.bidirectional = True
        self.hidden_size = 128
        self.embed_size = 128
        self.output_size = 128
        self.num_heads = 4
        self.max_grad_norm = 5.0  # The maximum gradient norm
        self.dropout = 0.2
        self.learning_rate = 0.001
        self.weight_decay = 0.0001
        self.shuffle = True  # Shuffle training set or not
        self.start_iteration = 0
        self.cuda = True
        self.lr_degrade_gamma = 0.5
        self.lr_degrade_step = 5
        self.file_suffix = f"{self.model_name}_{self.model_settings}" if self.model_settings is not None else f"{self.model_name}"

        # Trajectory pretraining

        self.temperature = 0.07
        self.checkpoint = os.path.join(self.root_dir, f"data/{self.dataset_name}/{self.model_name}_checkpoint.pt")
        self.bestpoint = os.path.join(self.root_dir, f"data/{self.dataset_name}/{self.model_name}_best.pt")

        # Finetune, finetune size set to be 10000
        self.task = "similarity learning"

        self.kseg = 5
        self.finetune = True
        self.load_pretrain = False
        self.finetune_print_freq = 10
        self.finetune_save_freq = 5
        self.finetune_learning_rate = 0.001
        self.finetune_learning_rate_decay = 0.0001

        self.similarity_learning_checkpoint = os.path.join(self.root_dir, f"data/{self.dataset_name}/{self.file_suffix}_{self.distance_type}_checkpoint.pt")
        self.similarity_learning_bestpoint = os.path.join(self.root_dir, f"data/{self.dataset_name}/{self.file_suffix}_{self.distance_type}_best.pt")


    def config_dataset(self):
        if self.dataset_name == "porto":
            # Boundary setting
            self.min_lon = -8.6868
            self.min_lat = 41.1128
            self.max_lon = -8.5611
            self.max_lat = 41.1928

            # Training setting
            self.num_train = 200000
            self.num_val = 10000
            self.num_fine_tune = 100000  # used for generate finetune ground truth
            self.num_finetune = 10000
            self.batch_size = 64
            self.best_threshold = 10
            self.avg_ti = 15.0  # avg time interval

            # Exp setting
            self.test_start = 800000
            self.num_test = 10000

        elif self.dataset_name == "tdrive":
            # Boundary setting
            self.min_lon = 116.2289
            self.min_lat = 39.8275
            self.max_lon = 116.4749
            self.max_lat = 40.0013

            # Training setting
            self.num_train = 10000
            self.num_val = 2000
            self.num_fine_tune = 10000  # used for generate finetune ground truth
            self.num_finetune = 10000
            self.batch_size = 64
            self.best_threshold = 5
            self.avg_ti = 165.8  # avg time interval

            # Exp setting
            self.test_start = self.num_train + self.num_val + self.num_finetune
            self.num_test = 2000

        elif self.dataset_name == "chengdu":
            # Boundary setting
            self.min_lon = 104.04214
            self.min_lat = 30.65294
            self.max_lon = 104.12958
            self.max_lat = 30.72775

            # Training setting
            self.num_train = 8000  # 200000
            self.num_val = 1000  # 10000
            self.num_test = 1000  # 10000
            self.batch_size = 64
            self.best_threshold = 10
            self.avg_ti = 3.3  # avg time interval

            # Exp setting
            self.test_start = self.num_train + self.num_val
            self.num_test = 2000


        else:
            print("Unknown dataset!")

        self.shuffle_node_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_node_list.npy")
        self.shuffle_rs_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_rs_list.npy")
        self.shuffle_time_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_time_list.npy")
        self.shuffle_st_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_st_list.npy")
        self.shuffle_d2vec_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_d2vec_list.npy")
        self.shuffle_coor_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_coor_list.npy")
        self.shuffle_kseg_file = os.path.join(self.root_dir, f"data/{self.dataset_name}/shuffle_kseg_list.npy")

        self.path_node_sets = os.path.join(self.root_dir,
                                           f"data/{self.dataset_name}/set/{self.distance_type}/node_sets_{self.num_companion}")
        self.path_rs_sets = os.path.join(self.root_dir,
                                         f"data/{self.dataset_name}/set/{self.distance_type}/rs_sets_{self.num_companion}")
        self.path_time_sets = os.path.join(self.root_dir,
                                           f"data/{self.dataset_name}/set/{self.distance_type}/time_sets_{self.num_companion}")
        self.path_st_sets = os.path.join(self.root_dir,
                                         f"data/{self.dataset_name}/set/{self.distance_type}/st_sets_{self.num_companion}")
        self.path_d2vec_sets = os.path.join(self.root_dir,
                                            f"data/{self.dataset_name}/set/{self.distance_type}/d2vec_sets_{self.num_companion}")
        self.path_index_sets = os.path.join(self.root_dir,
                                            f"data/{self.dataset_name}/set/{self.distance_type}/index_sets_{self.num_companion}")
        self.path_sets_truth = os.path.join(self.root_dir,
                                            f"data/{self.dataset_name}/ground_truth/{self.distance_type}/train_set_{self.num_companion}.npy")

        self.path_vali_truth = os.path.join(self.root_dir,
                                            f"data/{self.dataset_name}/ground_truth/{self.distance_type}/vali_st_distance.npy")
        self.path_test_truth = os.path.join(self.root_dir,
                                            f"data/{self.dataset_name}/ground_truth/{self.distance_type}/test_st_distance.npy")

    def config_train(self):
        # Segment Embedding
        with open(f"{self.root_dir}/data/{self.dataset_name}/{self.dataset_name}_features_param.pkl", "rb") as f:
            params = pickle.load(f)
            self.num_cls_token = params["num_cls_token"]
            self.num_loc_token = params["num_loc_token"]
            self.num_length_token = params["num_length_token"]
            self.num_radian_token = params["num_radian_token"]

    def config_device(self):
        if self.cuda and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

    def dataset_update(self, params):
        for param, value in params.items():
            if param in self.__dict__:
                setattr(self, param, value)
        self.config_dataset()

    def default_update(self, params=None):
        if isinstance(params, dict):
            for param, value in params.items():
                if param in self.__dict__:
                    setattr(self, param, value)
        self.config_dataset()
        self.config_device()
        self.config_train()



