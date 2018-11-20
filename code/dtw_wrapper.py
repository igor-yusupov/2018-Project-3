import numpy as np
import dill

from os.path import exists
from time import time, sleep
from threading import Thread

class DtwWrapper:
    num_th = 4

    def __init__(self, items, items_hash, dtw_function, distance_function, dtw_args={}, ch_num=1):
        self.items = items
        self.chanel_num = ch_num
        str_args = "".join(["{}{}".format(key, dtw_args[key]) for key in dtw_args])
        self.dtw_name = "{0}{1}{2}{3}{4}".format(dtw_function.__name__, dtw_function.__name__, str_args, items_hash, ch_num)
        self.series_distance = self.dtw_dist(dtw_function, distance_function, dtw_args)
        self.shape = items[0].loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.shape
        self.X_reshaped = np.array([x.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.reshape(1, -1)[0] for x in self.items])
        if exists("../data/distances/{0}.csv".format(self.dtw_name)):
            self.distances = np.genfromtxt("../data/distances/{0}.csv".format(self.dtw_name))
        else:
            self.distances = np.full((len(items), len(items)), -1)

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
        
    def fill_distances(self, print_time=True):
        t = time()
        ths = [Thread(target=self.computer, args=(i, self.dist)) for i in range(4)]
        for th in ths:
            th.start()
            
        for i in range(1000):
            alive = True
            for th in ths:
                alive &= th.is_alive()
            if not alive:
                break
            sleep(5)
            if i == 1:
                self.dump()

        for th in ths:
            th.join()
        
        self.dump()
            
        print(time() - t)

    def computer(self, n, f):
        size = len(self.items)
        bound_n = lambda n: list(range(n * size // 4, (n + 1) * size // 4))
        indexes = bound_n(n)

        for i in indexes:
            for j in range(indexes[0], i + 1):
                self.distances[i, j] = f(i, j)


        if n + 1 < DtwWrapper.num_th:
            next_indexes = bound_n(n + 1)
            for i in next_indexes:
                for j in indexes:
                    self.distances[i, j] = f(i, j)

        if n == 3:
            prev_indexes = bound_n(0)
            for i in indexes:
                for j in prev_indexes:
                    self.distances[i, j] = f(i, j)

        if n < 2:
            prev_indexes = bound_n(n + 2)
            for i in prev_indexes:
                for j in indexes:
                    self.distances[i, j] = f(i, j)
        
        return
