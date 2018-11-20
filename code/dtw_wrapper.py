import numpy as np
import dill

from os.path import exists

class DtwWrapper:
    def __init__(self, items, items_hash, dtw_function, distance_function, dtw_args={}, ch_num=1):
        self.items = items
        self.chanel_num = ch_num
        self.dtw_name = "{0}{1}{2}{3}".format(dtw_function.__name__, dtw_function.__name__, str(dtw_args), items_hash, ch_num)
        self.series_distance = self.dtw_dist(dtw_function, distance_function, dtw_args)
        self.shape = items[0].loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.shape
        self.X_reshaped = np.array([x.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.reshape(1, -1)[0] for x in self.X])
        if exists("../data/distances/{0}.csv".format(self.dtw_name)):
            self.distances = np.genfromtxt("../data/distances/{0}.csv".format(self.dtw_name))
        else:
            self.distances = np.full(len(items), len(items), -1)

    def dist(self, x_index, y_index):
        if self.distances[x_index, y_index] == -1:
            if self.distances[y_index, x_index] != -1:
                dist = self.distances[y_index, x_index]
            else:
                dist = self.series_distance(
                    self.X_reshaped[x_index],
                    self.X_reshaped[y_index]
                )
            self.distances[x_index, y_index] = dist
            self.distances[y_index, x_index] = dist

        return self.distances[x_index, y_index]

    def dtw_dist(self, dtw_function, distance_function, dtw_args):
        return lambda x, y: (dtw_function(x.reshape(self.shape), y.reshape(self.shape), distance_function, **dtw_args)[0])

    def dump(self):
        np.savetxt("../data/distances/{0}.csv".format(self.dtw_name), X=self.distances)
