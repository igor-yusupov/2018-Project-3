import numpy as np
import dill

from os.path import exists
from time import time, sleep
from threading import Thread, Lock

class DtwWrapper:
    """Wrapper for dtw functions
    Number of series must divide by 4

    """
    num_th = 4

    def __init__(self, items, items_hash, dtw_function, distance_function, dtw_args={}, ch_num=1):
        self.items = items
        self.chanel_num = ch_num
        str_args = "".join(["{}{}".format(key, dtw_args[key]) for key in dtw_args])
        self.dtw_name = "{0}{1}{2}{3}{4}".format(dtw_function.__name__, distance_function.__name__, str_args, items_hash, ch_num)
        self.series_distance = self.dtw_dist(dtw_function, distance_function, dtw_args)
        self.shape = items[0].loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.shape
        self.X_reshaped = np.array([x.loc[:, "ECoG_ch1":"ECoG_ch{0}".format(self.chanel_num)].values.reshape(1, -1)[0] for x in self.items])
        if exists("../data/distances/{0}.csv".format(self.dtw_name)):
            print("Loaded")
            self.distances = np.genfromtxt("../data/distances/{0}.csv".format(self.dtw_name))
        else:
            self.distances = np.full((len(items), len(items)), -1., dtype=float)

    def dist(self, x_index, y_index):
        x_index = int(x_index)
        y_index = int(y_index)
        if self.distances[x_index, y_index] != -1:
            return self.distances[x_index, y_index]

        self.distances[x_index, y_index] = self.series_distance(
            self.X_reshaped[x_index],
            self.X_reshaped[y_index]
        )

        return self.distances[x_index, y_index]

    def dtw_dist(self, dtw_function, distance_function, dtw_args):
        return lambda x, y: (dtw_function(x.reshape(self.shape), y.reshape(self.shape), distance_function, **dtw_args)[0])

    def dump(self):
        np.savetxt("../data/distances/{0}.csv".format(self.dtw_name), X=self.distances)
        
    def fill_distances(self, n_threads=4, print_time=True):
        t = time()
        ths = [Thread(target=self.computer, args=(i, self.dist, n_threads)) for i in range(n_threads)]
        for th in ths:
            th.start()
            
        for i in range(1000):
            alive = True
            for th in ths:
                alive &= th.is_alive()
            if not alive:
                break
            print("dump")
            self.dump()
            sleep(10)

        for th in ths:
            th.join()
        
        self.dump()    
        print(time() - t)

    def computer(self, n, f, n_threads):
        size = len(self.items)
        short_size = size // n_threads

        for i in range(size):
            for j in range(n * short_size, (n + 1) * short_size):
                self.distances[i, j] = f(i, j)

        return
