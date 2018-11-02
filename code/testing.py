import datetime
import dill
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from data_processing import DataIterator
from models import Autoregression

default_params = {
    "nrow": 100000,
    "window_size": 10,
    "element_length": 50,
    "path": "../data/Eye-Motion/ECoG.csv",
    "overlap": 10,
    "shuffle": True,
    "sample_size": 30,
    "chanel_num": 32,
    "repeat_num": 1
}


class TestFactory:

    def __init__(self, params=default_params, random_state=-1, it=None):
        self.data_path = params["path"]
        self.window_size = params["window_size"]
        self.element_length = params["element_length"]
        self.standart_sample_size = params["sample_size"]
        data = pd.read_csv(params["path"], header=0, nrows=params["nrow"])
        if it != None:
            self.data_iterator = it
        else:
            self.data_iterator = DataIterator(data, self.element_length, params["shuffle"], random_state=random_state)
        self.chanel_num = params["chanel_num"]
        if type(self.data_iterator) is DataIterator:
            self.shape = next(self.data_iterator).loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.shape
        else:
            self.shape = next(self.data_iterator)[0].loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.shape
        self.repeat_num = params["repeat_num"]
        self.results = []
        self.X = None
        self.classification_label = None
        counter = 0
        folders = os.listdir("../data/results/")
        while str(counter) in folders:
            counter += 1
        self.res_dir = "../data/results/{0}".format(counter)
        os.mkdir("../data/results/{0}".format(counter))

    def get_n(self, n):
        if type(self.data_iterator) is DataIterator:
            return [next(self.data_iterator) for i in range(n)]
        else:
            items = []
            labels = []
            for i in range(n):
                item, label = next(self.data_iterator)
                items.append(item)
                labels.append(label)

            return items, np.array(labels)

    def test_dtw(self, dtw_function, distance_function, description=None, sample_size=-1, dump_result=False, dtw_args={}):
        if sample_size < 0:
            sample_size = self.standart_sample_size

        if self.X is None:
            self.set_sample(sample_size)

        X_reshaped = np.array(
            [x.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.reshape(1, -1)[0] for x in self.X])

        start_time = time.time()
        for i in range(self.repeat_num):
            Z = linkage(X_reshaped, "average", metric=self.dtw_dist(dtw_function, distance_function, dtw_args))
        end_time = time.time()
        t = "{0:.3f}".format((end_time - start_time) / self.repeat_num)
        print("Elapsed time: {0}".format(t))
        info = ClusteredInfoDTW(self.X, Z, self.element_length, self.dtw(dtw_function, distance_function, dtw_args),
                                self.chanel_num, label=self.classification_label)
        if dump_result:
            with open("{0}/{1}".format(self.res_dir, description), "wb") as f:
                dill.dump(info, f)

        self.results.append((description, datetime.datetime.now(), t))

        return info

    def ar_clustering(self, window_size=10, dump_result=False, description=None, normalization=False):
        start_time = time.time()
        ar_models = []
        for (i, x) in enumerate(self.X):
            ar = Autoregression(x.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)], window_size, normalization)
            ar.fit()
            ar_models.append(ar.coeffecients())
            print("Trained: {0}".format(i + 1))

        coeffs = []
        for model in ar_models:
            coeff = [np.array(np.concatenate([x.detach().numpy() for x in chanel], axis=0)) for chanel in model]
            coeffs.append(np.concatenate(coeff, axis=0))
        coeffs = np.array(coeffs)
        end_time = time.time()
        Z = linkage(coeffs, "ward")
        t = "{0:.3f}".format((end_time - start_time) / self.repeat_num)
        print("Elapsed time: {0}".format(t))
        info = ClusteredInfoAR(self.X, Z, self.element_length, coeffs, self.chanel_num, label=self.classification_label)
        if dump_result:
            with open("{0}/{1}".format(self.res_dir, description), "wb") as f:
                dill.dump(info, f)

        return info

    def dtw_dist(self, dtw_function, distance_function, dtw_args):
        return lambda x, y: (dtw_function(x.reshape(self.shape), y.reshape(self.shape), distance_function, **dtw_args)[0])

    def dtw(self, dtw_function, distance_function, dtw_args):
        return lambda x, y: (dtw_function(x, y, distance_function, **dtw_args))

    def dump_result(self):
        with open("{0}/results.pkl".format(self.res_dir), "wb") as f:
            dill.dump(self.results, f)

    @staticmethod
    def load(file):
        with open(file, "rb") as f:
            return dill.load(f)

    def set_sample(self, n=-1):
        if type(self.data_iterator) is DataIterator:
            if n > 0:
                self.X = self.get_n(n)
            return self.X
        else:
            items, labels = self.get_n(n)
            self.X = items
            self.classification_label = labels
            return items, labels


class ClusteredInfo:

    def __init__(self, X, Z, element_length, chanel_num, description=None, label=None):
        self.X = X
        self.Z = Z
        self.element_length = element_length
        self.description = description
        self.count = len(X)
        self.chanel_num = chanel_num
        if label is not None:
            self.label = label
    
    @staticmethod
    def load(f):
        with open(f, "rb") as f:
            return dill.load(f)

    def cluster(self, cluster_num):
        self.cluster_num = cluster_num
        self.clusters_labels = fcluster(self.Z, cluster_num, criterion="maxclust")
        unique_elements, counts_elements = np.unique(self.clusters_labels, return_counts=True)
        self.stats = pd.DataFrame(index=unique_elements, data=counts_elements, columns=["count"])
        self.stats = self.stats.sort_values(by="count", ascending=False)

    def visualize(self):
        fig = plt.figure(figsize=(25, 10))
        dn = dendrogram(self.Z)
    
    def show_chanel(self, ch=1, label=1, max_num=5):
        idxs = np.where(self.clusters_labels == label)[0]
        time = np.linspace(0, self.element_length - 1, self.element_length)
        if max_num is None:
            num_graphs = len(idxs)
        else:
            num_graphs = np.min([max_num, len(idxs)])
        fig, ax = plt.subplots(num_graphs, 1, sharex=True, squeeze=False, figsize=(15, 2.5 * num_graphs),
                            constrained_layout=False)
        fig.suptitle("Chanel {0} of {1} cluster".format(ch, label), y=1, fontsize = 14);
        for i in range(num_graphs):
            ax[i][0].plot(time, self.X[idxs[i]].loc[:, "ECoG_ch{}".format(ch)]);
            ax[i][0].grid()
                
        plt.tight_layout();

    def clusters_compare_table(self, label, num_series=5, max_chanels=5):
        idxs = np.where(self.clusters_labels == label)[0]
        num_series = min(len(idxs), num_series)
        chanels = min(self.chanel_num, max_chanels)
        if num_series == 0:
            return

        fig, ax = plt.subplots(num_series, chanels, sharex=True, squeeze=False, figsize=(20, 2.5 * num_series), constrained_layout=False)
        fig.suptitle("Cluster {0}".format(label), y=1.01, fontsize = 14)
        
        t = self.X[0].loc[:, "ECoG_time"]
        for df_id in range(num_series):
            for ch in range(1, chanels + 1):
                x = self.X[idxs[df_id]].loc[:, "ECoG_ch{0}".format(ch)].values
                ax[df_id][ch - 1].plot(t, x)
                if df_id == 0:
                    ax[df_id][ch - 1].set_xlabel("ch{}".format(ch), fontsize=14)
                if ch == 1:
                    ax[df_id][ch - 1].set_ylabel("ts{}".format(df_id + 1),fontsize=14)
                ax[df_id][ch - 1].xaxis.set_label_position("top")
                ax[df_id][ch - 1].xaxis.label.set_color("red")
                ax[df_id][ch - 1].yaxis.label.set_color("red")
        
        plt.tight_layout();

    def compating_at_one(self, label, num_series=5, max_chanel=5):
        idxs = np.where(self.clusters_labels == label)[0]
        num_series = min(len(idxs), num_series)
        if num_series == 0:
            return
        showed_chanels_num = min(self.chanel_num, max_chanel)
        if showed_chanels_num == 0:
            return

        fig, axs = plt.subplots(showed_chanels_num, 1, sharex=True, squeeze=False, figsize=(15, 2.5 * showed_chanels_num), constrained_layout=False)
        fig.suptitle("Cluster {0}".format(label), y=1.01, fontsize = 14)

        for i in range(num_series):
            df_id = idxs[i]
            for (chanel_id, ax) in enumerate(axs):
                ax[0].set_title("Chanel {0}".format(chanel_id))
                y = self.X[df_id].loc[:, "ECoG_ch{0}".format(chanel_id + 1)].values
                ax[0].plot(y, label="y ts:{0}".format(df_id))

        plt.tight_layout();


class ClusteredInfoAR(ClusteredInfo):
    def __init__(self, X, Z, element_length, coeffs, chanel_num, description=None, label=None):
        ClusteredInfo.__init__(self, X, Z, element_length, chanel_num, description, label)
        self.coeffs = coeffs


class ClusteredInfoDTW(ClusteredInfo):

    def __init__(self, X, Z, element_length, dtw, chanel_num, description=None, label=None):
        ClusteredInfo.__init__(self, X, Z, element_length, chanel_num, description, label)
        self.paths = dict()
        self.dtw = dtw
  
    def allignment_to_random(self, label, num_series=5, max_chanel=5, show_real_y=False):
        idxs = np.where(self.clusters_labels == label)[0]
        num_series = min(len(idxs), num_series)
        if num_series == 0:
            return
        showed_chanels_num = min(self.chanel_num, max_chanel)
        if showed_chanels_num == 0:
            return
        align_to_id = np.random.choice(idxs)
        x_fixes = self.X[align_to_id]

        fig, axs = plt.subplots(showed_chanels_num, 1, sharex=True, squeeze=False, figsize=(15, 2.5 * showed_chanels_num), constrained_layout=False)
        fig.suptitle("Cluster {0}".format(label), y=1.01, fontsize = 14)

        for (chanel_id, ax) in enumerate(axs):
            x = x_fixes.loc[:, "ECoG_ch{0}".format(chanel_id + 1)].values
            ax[0].plot(x, "black", label="X", linewidth=3)

        for i in range(num_series):
            df_id = idxs[i]
            if df_id == align_to_id:
                continue
            if align_to_id in self.paths and df_id in self.paths[align_to_id]:
                path = self.paths[align_to_id][df_id]
            elif df_id in self.paths and align_to_id in self.paths[df_id]:
                path = (self.paths[df_id][align_to_id][1], self.paths[df_id][align_to_id][0])
            else:
                if df_id not in self.paths:
                    self.paths[df_id] = dict()
                path = self.dtw(x_fixes.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values, 
                    self.X[df_id].loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values)[3]
                self.paths[df_id][align_to_id] = path

            for (chanel_id, ax) in enumerate(axs):
                y = self.X[df_id].loc[:, "ECoG_ch{0}".format(chanel_id + 1)].values
                y_new = pd.DataFrame([y[i] for i in path[1]], index=path[0])
                y_new = y_new.groupby(y_new.index).mean().values.reshape(-1)
                ax[0].plot(y_new, label="y_new ts:{0}".format(df_id))
                if show_real_y:
                    ax[0].plot(y, label="y ts:{0}".format(df_id))
                    for (map_x, map_y) in np.array(path).transpose():
                        plt.plot([map_x, map_y], [x[map_x], y[map_y]], "black", linewidth=0.3)
                # ax[0].legend(loc=1)

        plt.tight_layout();
