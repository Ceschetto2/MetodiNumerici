import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegression:
    def __init__(self):
        self.y = None
        self.X = None
        self.B = None
        self.W = None



    def fit(self, X, y, n_iterations = 10, atol = 1e-6, rtol = 1e-6):


        n_samples, n_features = X.shape
        self.X = np.hstack((np.ones((n_samples, 1)), X))  # (n, p+1)
        y = y.reshape(-1, 1)  # (n, 1)
        B_new = B_old = np.zeros((self.X.shape[1], 1))  # (p+1, 1)

        for i in range(n_iterations):
            linear_prediction = self.X @ B_old

            p = sigmoid(linear_prediction)  # (n, 1)

            W = np.diagflat((p * (1 - p)))  # (n x n)
            grad = self.X.T @ (p - y)
            H = self.X.T @ W @ self.X

            B_new = B_old - np.linalg.pinv(H)@grad
            if np.linalg.norm(B_new - B_old) < atol + rtol * np.linalg.norm(B_old):
                break
            B_old = B_new
        self.B = B_new
        return self


    def predict(self, X_predict):
        X_predict = np.hstack((np.ones((X_predict.shape[0], 1)), X_predict))
        linear_prediction = X_predict @ self.B

        prediction = sigmoid(linear_prediction)
        y_pred = []
        for y in prediction:
            if y > 0.5:
                y_pred.append(1)
            elif y < 0.5:
                y_pred.append(0)
            else:
                y_pred.append(np.random.choice(2))

        return y_pred

