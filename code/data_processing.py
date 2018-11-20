import numpy as np
import pandas as pd

from collections import OrderedDict


class DataIterator(StopIteration):

    def __init__(self, data: pd.DataFrame, element_length, shuffle=True, overlap=0, random_state=None):
        self.pseudo_random_gen = np.random.RandomState(random_state)
        self.data = data
        self.element_length = element_length
        unique_element_length = element_length - overlap
        sample_size = (data.shape[0] - overlap) // unique_element_length
        self.starts = np.linspace(0, (sample_size - 1) * unique_element_length, sample_size, dtype=int)
        self.position = 0
        if shuffle:
            self.pseudo_random_gen.shuffle(self.starts)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position < self.starts.shape[0]:
            start = self.starts[self.position]
            self.position += 1
            return self.data.loc[start:start + self.element_length - 1, :]
        raise StopIteration


class OneDimDataIterator(StopIteration):
    def __init__(self, data: pd.DataFrame, element_length, shuffle=True, overlap=0, random_state=0):
        self.pseudo_random_gen = np.random.RandomState(random_state)
        self.data = data
        self.element_length = element_length
        unique_element_length = element_length - overlap
        sample_size = (data.shape[1] - overlap) // unique_element_length
        self.indexes = data.index.values.copy()
        self.position = 0
        self.labels = data.iloc[:, 0]
        self.labeled_indexes = dict()
        self.labeled_indexes_iter = dict()
        self.starts = dict()
        for i in self.indexes:
            self.starts[i] = list(np.linspace(0, (sample_size - 1) * unique_element_length, sample_size, dtype=int))
            self.pseudo_random_gen.shuffle(self.starts[i])
        for label in self.labels.unique():
            indexes = list(self.labels[self.labels == 4].index.values)
            self.pseudo_random_gen.shuffle(indexes)
            self.labeled_indexes[label] = OrderedDict.fromkeys(indexes)
            for series in self.labeled_indexes[label]:
                self.labeled_indexes[label][series] = \
                    list(np.linspace(0, (sample_size - 1) * unique_element_length, sample_size, dtype=int))
                self.pseudo_random_gen.shuffle(self.labeled_indexes[label][series])

            self.labeled_indexes_iter[label] = iter(list(self.labeled_indexes[label].keys()))

        if shuffle:
            self.pseudo_random_gen.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        class_label = self.pseudo_random_gen.choice(list(self.labeled_indexes.keys()))

        try:
            cur_index = next(self.labeled_indexes_iter[class_label])
            start = self.labeled_indexes[class_label][cur_index].pop()
            ret = pd.DataFrame({
                "ECoG_time": pd.RangeIndex(0, self.element_length),
                "ECoG_ch1": self.data.loc[
                            cur_index,
                            start: start + self.element_length - 1]})

            if len(self.labeled_indexes[class_label][cur_index]) == 0:
                del self.labeled_indexes[class_label][cur_index]

            return ret, class_label, (cur_index, start)

        except StopIteration:
            if len(self.labeled_indexes[class_label]) == 0:
                raise StopIteration

            self.labeled_indexes_iter[class_label] = iter(list(self.labeled_indexes[class_label].keys()))
            return next(self)


