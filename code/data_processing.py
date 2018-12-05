import numpy as np
import pandas as pd

from collections import OrderedDict


class DataIterator(StopIteration):

    def __init__(self, data: pd.DataFrame, element_length, shuffle=True, overlap=0, random_state=None, labeled=False):
        self.pseudo_random_gen = np.random.RandomState(random_state)
        self.data = data.set_index("obj")
        self.element_length = element_length
        self.labeled = labeled
        self.obj_indexes = dict()
        self.labels = data.label.unique()
        for label in self.labels:
            self.obj_indexes[label] = list(data[data.label == label].obj.unique())
            if shuffle:
                self.pseudo_random_gen.shuffle(self.obj_indexes[label])

    def __iter__(self):
        return self

    def __next__(self):
        try:
            label = self.pseudo_random_gen.choice(self.labels)
            index = self.obj_indexes[label].pop()
            
            return self.data.loc[index, "0":].values.T, label, index
        except:
            raise StopIteration
