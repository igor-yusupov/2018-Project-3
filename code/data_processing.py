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
        shuffle_set(self.indexes)
        if shuffle:
            if random_state != -1:
                np.random.shuffle(self.indexes, random_state=random_state)
            else:
                np.random.shuffle(self.indexes)

    def __iter__(self):
        return self

    def __next__(self):
        if self.position < len(self.indexes) - 1:
            start = np.random.randint(1, 599 - self.element_length)
            self.position += 1
            ret = pd.DataFrame({"ECoG_time" : pd.RangeIndex(0, self.element_length),
                                "ECoG_ch1": self.data.loc[self.indexes[self.position], start:start + self.element_length - 1]})

            return ret, self.labels[self.indexes[self.position]]

        raise StopIteration

