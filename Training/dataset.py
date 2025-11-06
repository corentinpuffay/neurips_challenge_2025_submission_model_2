import os
import concurrent.futures
import numpy as np
import pandas
from torch.utils.data import Dataset
import mat73

class Dataset(Dataset):
    def __init__(self, cachepath):
        data_dict = mat73.loadmat(cachepath)

        self.__nb_caches = int(data_dict['cacheNumber'])
        self.__cache_folder = data_dict['cacheFolder']
        self.__cache_name = data_dict['cacheName']
        self.__nb_batches = int(data_dict['nbBatches'])
        self.__fs = int(data_dict['fs'])
        self.__nb_channels = int(data_dict['nbChannels'])

        # Load first cache to get batches, labels
        self.__executor = concurrent.futures.ThreadPoolExecutor()
        self.__next_cache_index = 1
        self.__next = self.__load_caches_background(self.__next_cache_index)
        self.__batches, self.__labels = self.__next.result()
        self.__next = None
        self.__next_cache_index += 1
        self.__offset = 0

    def __load_caches_background(self, cache_index):
        return self.__executor.submit(self.__load_caches, cache_index)

    def __load_parquet(self, filepath):
        d = pandas.read_parquet(filepath).to_numpy()
        batches = np.zeros((d.shape[0] // self.__nb_channels, self.__nb_channels, d.shape[1] - 3), dtype=np.float32)  # (batches, channels, time)
        labels = np.zeros(d.shape[0] // self.__nb_channels, dtype=np.float32)  # (batches, )
        for i in range(d.shape[0] // self.__nb_channels):
            batch_data = d[i * self.__nb_channels:(i + 1) * self.__nb_channels, 3:]
            batch_label = d[i * self.__nb_channels, 2]
            batches[i, :, :] = batch_data
            labels[i] = batch_label

        # channel_mask = np.array([127, 126, 49, 113, 68, 94, 11, 62, 40, 109]) - 1
        # batches = batches[:, channel_mask, :]

        return batches, labels

    def __load_caches(self, cache_index):
        assert cache_index <= self.__nb_caches
        current_cache_path = os.path.join(self.__cache_folder, f"{self.__cache_name}_{cache_index}.parquet")
        batches, labels = self.__load_parquet(current_cache_path)

        ## Randomise the order of the batches and labels in the same way
        perm = np.random.permutation(batches.shape[0])
        batches = batches[perm, :, :]
        labels = labels[perm]

        return batches, labels

    def __len__(self):
        return self.__nb_batches

    def fs(self):
        return self.__fs

    def __getitem__(self, index):
        index = index - self.__offset

        if self.__next is None and self.__nb_caches > 1: # start loading next cache in background
            if self.__next_cache_index > self.__nb_caches:
                self.__next_cache_index = 1
            self.__next = self.__load_caches_background(self.__next_cache_index)
            self.__next_cache_index += 1

        if index >= self.__batches.shape[0]:  # need to load next cache
            self.__offset += self.__batches.shape[0]
            self.__batches, self.__labels = self.__next.result()
            self.__next = None
            index = 0

        if self.__nb_caches > 1 and (index + self.__offset) == (self.__nb_batches - 1):
            self.__offset = 0
            self.__batches, self.__labels = self.__next.result()
            self.__next = None
            index = 0

        batch = self.__batches[index, :, :]
        label = self.__labels[index]

        return batch, label