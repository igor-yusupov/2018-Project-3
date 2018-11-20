from random import choice

import numpy as np
import pandas as pd

from sklearn.utils import shuffle as shuffle_set


class DataIterator(StopIteration):

    def __init__(self, data: pd.DataFrame, element_length, shuffle=True, overlap=0, random_state=-1):
        self.data = data
        self.element_length = element_length
        unique_element_length = element_length - overlap
        sample_size = (data.shape[0] - overlap) // unique_element_length
        self.starts = np.linspace(0, (sample_size - 1) * unique_element_length, sample_size, dtype=int)
        self.position = 0
        self.random_state = random_state
        if shuffle:
            if random_state != -1:
                shuffle_set(self.starts, random_state=random_state)
            else:
                shuffle_set(self.starts)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position < self.starts.shape[0]:
            start = self.starts[self.position]
            self.position += 1
            return self.data.loc[start:start + self.element_length - 1, :]
        raise StopIteration


class OneDimDataIterator(StopIteration):
    def __init__(self, data: pd.DataFrame, element_length, shuffle=True, overlap=0, random_state=-1):
        self.data = data
        self.element_length = element_length
        unique_element_length = element_length - overlap
        sample_size = (data.shape[0] - overlap) // unique_element_length
        self.indexes = data.index.values.copy()
        self.position = 0
        self.random_state = random_state
        self.labels = data.iloc[:, 0]
        self.labeled_indexes = dict()
        self.labeled_positions = dict()
        for label in self.labels.unique():
            self.labeled_indexes[label] = self.labels[self.labels == 4].index.values.copy()
            self.labeled_positions[label] = 0
            if shuffle:
                if random_state != -1:
                    np.random.shuffle(self.labeled_indexes[label], random_state=random_state)
                else:
                    np.random.shuffle(self.labeled_indexes[label])
        shuffle_set(self.indexes)
        if shuffle:
            if random_state != -1:
                np.random.shuffle(self.indexes, random_state=random_state)
            else:
                np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        class_label = choice(list(self.labeled_indexes.keys()))
        if class_label is not None:
            if self.labeled_positions[class_label] < len(self.labeled_indexes[class_label]) - 1:
                start = np.random.randint(1, 599 - self.element_length)
                self.labeled_positions[class_label] += 1
                ret = pd.DataFrame({"ECoG_time": pd.RangeIndex(0, self.element_length),
                                    "ECoG_ch1": self.data.loc[self.labeled_indexes[class_label][self.labeled_positions[class_label]],
                                                start:start + self.element_length - 1]})

                return ret, class_label

        raise StopIteration
