import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

import config
from accept_reject_learner import PointAccepter


class OnlineRegression:
    def __init__(self, X, y, thresh=config.CHANGE_THRESH):
        self.X = X
        self.y = y
        self.X_train = [X[0]]
        self.y_train = [y[0]]

        self.reg = linear_model.LinearRegression(fit_intercept=False)
        self.reg.fit(self.X_train, self.y_train)

        self.accepter = PointAccepter()
        self.thresh = thresh

    def fit(self):
        for i in range(1, self.X.shape[0]):
            p = self.accepter.pred(self.X[i])
            if p > 0.5:
                self.X_train.append(self.X[i])
                self.y_train.append(self.y[i])
                old_coeffs = self.reg.coef_
                self.reg.fit(
                    np.array(self.X_train),
                    np.array(self.y_train))
                delta = np.linalg.norm(
                    self.reg.coef_ - old_coeffs
                ) / config.INPUT_DIM
                self.accepter.fit(
                    self.X[i],
                    delta > self.thresh
            )

        y_pred = self.reg.predict(self.X)

        reg_rand = linear_model.LinearRegression(fit_intercept=False)
        rand_inds = np.random.choice(range(self.X.shape[0]), len(self.X_train))
        reg_rand.fit(self.X[rand_inds], self.y[rand_inds])
        y_rand = reg_rand.predict(self.X)

        return (
            float(len(self.X_train)/len(self.X)),
            mean_squared_error(self.y, y_pred),
            mean_squared_error(self.y, y_rand)
        )


if True or __name__ == '__main__':

    diabetes = datasets.load_diabetes()
    data_X = diabetes.data - np.mean(diabetes.data)
    data_y = diabetes.target - np.mean(diabetes.target)
    p_vs_mse = []
    for t in np.arange(0, 30, 0.5):
        r = OnlineRegression(data_X, data_y, t)
        p_vs_mse. append(tuple(r.fit()))
    plt.scatter(*zip(*p_vs_mse))
    plt.xlabel('Proportion of Points Used')
    plt.ylabel('MSE')
    plt.show()

