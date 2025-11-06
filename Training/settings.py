import os
import dataclasses

@dataclasses.dataclass
class Settings:

    def __init__(self, challenge=None):
        self.__batch_size = 2048
        self.__accumulation_steps = 4
        self.__d_model = 16 # 256, 64 or 16
        self.__dim_feedforward = 2 * self.__d_model # 4 or 2
        self.__num_layers = 1 # 6 or 3
        self.__nheads = 1 # 8 or 4
        self.__dropout = 0.3 # 0.3 or 0.1
        self.__huber_delta = 1.0 # 0.5, 1.0 or 2.0
        self.__n_epochs = 100 # Fixed
        self.__loss_fn = 'huber'  # 'huber' or 'NRMSE'
        self.__transformer = 'convolutional_stem' # TODO
        self.number = -1  # just for logging purposes
        self.challenge = challenge

    def cache_folder(self):
        if self.challenge == 1:
            return '/Volumes/GREY SSD/cache/scratch/2s/challenge1_parquet'
        elif self.challenge == 2:
            return '/Volumes/GREY SSD/cache/scratch/2s/NeurIPS_externalizing_parquet'
        else:
            raise NotImplementedError

    def log_path(self):
        if self.challenge == 1:
            return "sweeper_log_challenge1.txt"
        elif self.challenge == 2:
            return "sweeper_log_challenge2.txt"
        else:
            raise NotImplementedError

    def summary_paths(self):
        if self.challenge == 1:
            return "formatted_output_challenge1.txt", "summary_output_challenge1.txt"
        elif self.challenge == 2:
            return "formatted_output_challenge2.txt", "summary_output_challenge2.txt"
        else:
            raise NotImplementedError

    def ensemble_log_path(self):
        if self.challenge == 1:
            return "ensemble_log_challenge1.txt"
        elif self.challenge == 2:
            return "ensemble_log_challenge2.txt"
        else:
            raise NotImplementedError

    def ensemble_summary_path(self):
        if self.challenge == 1:
            return "ensemble_summary_challenge1.txt"
        elif self.challenge == 2:
            return "ensemble_summary_challenge2.txt"
        else:
            raise NotImplementedError

    def model_folder(self):
        if self.challenge == 1:
            return "/Volumes/GREY SSD/cache/scratch/models_challenge1"
        elif self.challenge == 2:
            return "/Volumes/GREY SSD/cache/scratch/models_challenge2"
        else:
            raise NotImplementedError

    def good_model_folder(self):
        if self.challenge == 1:
            return "/Volumes/GREY SSD/cache/scratch/good_models_challenge1"
        elif self.challenge == 2:
            return "/Volumes/GREY SSD/cache/scratch/good_models_challenge2"
        else:
            raise NotImplementedError

    def submission_model_folder(self):
        if self.challenge == 1:
            return "/Volumes/GREY SSD/cache/scratch/submission_models_challenge1"
        elif self.challenge == 2:
            return "/Volumes/GREY SSD/cache/scratch/submission_models_challenge2"
        else:
            raise NotImplementedError

    def training_path(self):
        return os.path.join(self.cache_folder(), 'R1_R2_R3_R4_R6_R7_R8_R9_contrastChangeDetection_training.mat')

    def validation1_path(self):
        return os.path.join(self.cache_folder(), 'R1_R2_R3_R4_R6_R7_R8_R9_contrastChangeDetection_validation.mat')

    def validation2_path(self):
        return os.path.join(self.cache_folder(), 'NC_contrastChangeDetection.mat')

    def test_path(self):
        return os.path.join(self.cache_folder(), 'R5_contrastChangeDetection.mat')

    @property
    def batch_size(self):
        return self.__batch_size

    @batch_size.setter
    def batch_size(self, value):
        self.__batch_size = value

    @property
    def accumulation_steps(self):
        return self.__accumulation_steps

    @accumulation_steps.setter
    def accumulation_steps(self, value):
        self.__accumulation_steps = value

    @property
    def d_model(self):
        return self.__d_model

    @d_model.setter
    def d_model(self, value):
        self.__d_model = value

    @property
    def dim_feedforward(self):
        return self.__dim_feedforward

    @dim_feedforward.setter
    def dim_feedforward(self, value):
        self.__dim_feedforward = value

    @property
    def num_layers(self):
        return self.__num_layers

    @num_layers.setter
    def num_layers(self, value):
        self.__num_layers = value

    @property
    def nheads(self):
        return self.__nheads

    @nheads.setter
    def nheads(self, value):
        self.__nheads = value

    @property
    def dropout(self):
        return self.__dropout

    @dropout.setter
    def dropout(self, value):
        self.__dropout = value

    @property
    def huber_delta(self):
        return self.__huber_delta

    @huber_delta.setter
    def huber_delta(self, value):
        self.__huber_delta = value

    @property
    def n_epochs(self):
        return self.__n_epochs

    @n_epochs.setter
    def n_epochs(self, value):
        self.__n_epochs = value

    @property
    def loss_fn(self):
        return self.__loss_fn

    @loss_fn.setter
    def loss_fn(self, value):
        self.__loss_fn = value

    def __str__(self):
        return (f"Settings(batch_size={self.batch_size}, "
                f"accumulation_steps={self.accumulation_steps}, "
                f"d_model={self.d_model}, "
                f"dim_feedforward={self.dim_feedforward}, "
                f"num_layers={self.num_layers}, "
                f"nheads={self.nheads}, "
                f"dropout={self.dropout}, "
                f"huber_delta={self.huber_delta}, "
                f"n_epochs={self.n_epochs}, "
                f"loss_fn='{self.loss_fn}')")


