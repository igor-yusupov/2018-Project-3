import pandas as pd
import numpy as np
import torch

class Autoregression:
    """Model calculates coefficients for autoregression model.

    Attributes:
        data (DataFrame): input time series in dataframe.
        X (torch.Tensor): prepared data for model.
        y (torch.Tensor): prepared target.
    """

    def __init__(self, data, window_size):
        self.data = data
        self.window_size = window_size
        self.chanel_num = data.loc[:, "ECoG_ch1":].shape[1]
        x = data.loc[:, "ECoG_ch1":].values
        X = np.array([x[predict-window_size : predict].reshape(x.shape[1], -1) for predict in range(window_size, x.shape[0], 1)])
        y = np.array([x[predict] for predict in range(window_size, x.shape[0], 1)])
        with torch.no_grad():
            self.X = torch.FloatTensor([X[:,i,:] for i in range(X.shape[1])])
            self.y = torch.FloatTensor(y)

        self.learnable_functions = []
        self.params = []
        for i in range(self.chanel_num):
            self.learnable_functions.append(torch.nn.Linear(window_size, 1))
            self.params.extend(list(self.learnable_functions[-1].parameters()))
        self.optimizer = torch.optim.Adagrad(self.params, lr=10.)
        self.norm = torch.nn.MSELoss()
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=700, gamma=0.99)

        self._predicted_series = None

    def fit(self, window_size):
        for i in range(30000):
            loss = 0
            for (chanel_id, fun) in enumerate(self.learnable_functions):
                out = fun(self.X[chanel_id])
                loss += self.norm(out.reshape(-1), self.y[:, chanel_id])
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.lr_scheduler.step()
            
            if i % 2000 == 0:
                print("Loss: {0:0.3f}".format(loss))

            del loss

    def coeffecients(self):
        """ Output model coefficients.

        Returns:
            `list` of (`np.array`, `np.array`): weights and constant in linear model.
        """
        results = []
        for i in range(len(self.params) // 2):
            results.append((self.params[2*i], self.params[2*i + 1]))

        return results

    def predicted_series(self):
        """ Predict time series.

        Returns:
            `list` of `np.array`
        """
        if self._predicted_series is not None:
            return self._predicted_series
        self._predicted_series = []
        x = self.data.loc[:, "ECoG_ch1" : "ECoG_ch5"].values
        for (ch_id, fun) in enumerate(self.learnable_functions):
            predicted_ch = fun(self.X[ch_id])
            self._predicted_series.append(np.concatenate([x[:self.window_size, ch_id], predicted_ch.detach().numpy().reshape(-1)]))
        
        return self._predicted_series
