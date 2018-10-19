import datetime
import pickle
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.cluster.hierarchy import linkage, dendrogram
from data_processing import DataIterator

default_params = {
    "nrow": 100000,
    "window_size": 10,
    "element_length": 300,
    "path": "../data/Eye-Motion/ECoG.csv",
    "overlap": 0,
    "shuffle": True,
    "sample_size": 10,
    "chanel_num": 4,
    "repeat_num": 1
}


class TestFactory:

    def __init__(self, params=default_params, random_state=-1):
        self.data_path = params["path"]
        self.window_size = params["window_size"]
        self.element_length = params["element_length"]
        self.standart_sample_size = params["sample_size"]
        data = pd.read_csv('../data/Eye-Motion/ECoG.csv', header=0, nrows=params["nrow"])
        self.data_iterator = DataIterator(data, self.element_length, params["shuffle"], random_state=random_state)
        self.chanel_num = params["chanel_num"]
        self.shape = next(self.data_iterator).loc[:, "ECoG_ch1":"ECoG_ch3"].values.shape
        self.repeat_num = params["repeat_num"]
        self.results = []
        self.X = None
        counter = 0
        folders = os.listdir("../data/results/")
        while str(counter) in folders:
            counter += 1
        self.res_dir = "../data/results/{0}".format(counter)
        os.mkdir("../data/results/{0}".format(counter))

    def get_n(self, n):
        return [next(self.data_iterator) for i in range(n)]

    def test_dtw(self, dtw_function, distance_function, sample_size=-1, visualize=False, dump_result=False,
                 dist_name=None, dtw_args={}):
        if sample_size < 0:
            sample_size = self.standart_sample_size

        if self.X is None:
            self.set_sample(sample_size)

        X_reshaped = np.array(
            [x.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num - 1)].values.reshape(1, -1)[0] for x in self.X])

        start_time = time.time()
        for i in range(self.repeat_num):
            Z = linkage(X_reshaped, metric=self.dtw_dist(dtw_function, distance_function, dtw_args))
        end_time = time.time()

        print("Elapsed time: {0:0.4}".format((end_time - start_time) / self.repeat_num))
        if visualize:
            self.visualize(Z)
        if dump_result:
            distance_name = distance_function.__name__ if dist_name is None else dist_name
            with open("{0}/{1}_{2}".format(self.res_dir, dtw_function.__name__, distance_name), 'wb') as f:
                pickle.dump(Z, f)

        self.results.append((dtw_function.__name__, distance_function.__name__, datetime.datetime.now(),
                             (end_time - start_time) / self.repeat_num))

        return Z

    def dtw_dist(self, dtw_function, distance_function, dtw_args):
        return lambda x, y: (dtw_function(x.reshape(self.shape), y.reshape(self.shape), distance_function, **dtw_args)[0])

    @staticmethod
    def visualize(Z):
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(Z)

    def dump_result(self):
        with open("{0}/results.pkl".format(self.res_dir), 'wb') as f:
            pickle.dump(self.results, f)

    @staticmethod
    def load(file):
        with open(file, 'rb') as f:
            return pickle.load(f)

    def set_sample(self, n=-1):
        if n > 0:
            self.X = self.get_n(n)
        return self.X

    def show_clustered(self, links, cluster_labels, ch=1, label=None, max_num=None):
        idxs = np.where(cluster_labels == label)[0]
        time = np.linspace(0, self.element_length - 1, self.element_length)
        if max_num is None:
            num_graphs = len(idxs)
        else:
            num_graphs = np.min([max_num, len(idxs)])
        fig, ax = plt.subplots(num_graphs, 1, sharex=True, squeeze=False, figsize=(14, 2.5 * num_graphs),
                            constrained_layout=False)
        fig.suptitle("Chanel {0} of {1} cluster".format(ch, label), y=1, fontsize = 14);
        for i in range(num_graphs):
            ax[i][0].plot(time, self.X[idxs[i]].loc[:, "ECoG_ch{}".format(ch)]);
            ax[i][0].grid()
                
        plt.tight_layout();
