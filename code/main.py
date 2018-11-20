import numpy as np
import pandas as pd

from dtw_wrapper import DtwWrapper
from data_processing import OneDimDataIterator
from testing import TestFactory
from dtw import dtw_ln

if __name__ == '__main__':

	data = pd.read_csv("../data/preprocessed_large.csv", header=None)

	params = {
		"nrow": 1000000,
		"window_size": 10,
		"element_length": 50,
		"path": "../data/preprocessed_large.csv",
		"overlap": 0,
		"shuffle": True,
		"sample_size": 10,
		"chanel_num": 1,
		"repeat_num": 1
	}

	it = OneDimDataIterator(data, params["element_length"])

	tests = TestFactory(params, random_state=42, it=it)
	X = tests.set_sample(40)
	dtw = DtwWrapper(X[0], hash(tests.infos), dtw_ln, lambda x, y: np.linalg.norm(x - y), dtw_args={"l": 0.2, "zscr": True})
	dtw.fill_distances()