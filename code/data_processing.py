import numpy as np
import pandas

from sklearn.utils import shuffle as shuffle_set


class DataIterator(StopIteration):

    def __init__(self, data: pandas.DataFrame, element_length, shuffle=True, overlap=0, random_state=-1):
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
