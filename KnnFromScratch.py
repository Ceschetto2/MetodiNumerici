from collections import Counter
from functools import partial
from sklearn.decomposition import TruncatedSVD

import numpy as np
import statistics

from matplotlib import pyplot as plt


def truncated_svd(X, k):

    XtX = X.T @ X

    eigvals, eigvecs = np.linalg.eigh(XtX)

    sorted_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_idx]
    eigvecs = eigvecs[:, sorted_idx]

    lambdas_k = eigvals[:k]
    V_k = eigvecs[:, :k]  # d × k
    sigma_k = np.sqrt(lambdas_k)  # (k,)

    U_k = np.zeros((X.shape[0], k))
    for i in range(k):
        if V_k[0, i] < 0:
            V_k[:, i] *= -1
        U_k[:, i] = (X @ V_k[:, i]) / sigma_k[i]
    sigma_k = np.diag(sigma_k)
    return U_k, sigma_k, V_k






###METRICHE
def  minkowski(a,b, p):
    pass
    return np.sum((np.abs(a-b))**p)**(1/p)

def cosine_similarity(a,b):
    pass
    return 1 - (np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))

def chebyshev(a,b):
    pass
    return np.abs(a-b).max()



###KNN
class KnnFromScratch:
    def __init__(self, k ):
        self.distance_function = None
        self.y = None
        self.X = None
        self.X_predict = None
        self.k = k



    def fit(self, X, y):
        self.X = X
        self.y = y
        return self

    def predict(self, X_predict, metric='minkowski', p = 2):
        self.X_predict=X_predict

        match metric:
            case "minkowski":
                self.distance_function = partial(minkowski, p = p)
            case "manhattan":
                self.distance_function = partial(minkowski, p = 1)
            case "euclidean":
                self.distance_function = partial(minkowski, p = 2)
            case "cosine":
                self.distance_function = cosine_similarity
            case "chebyshev":
                self.distance_function = chebyshev

        y_predict = [self._predict(x_predict) for x_predict in X_predict]
        return y_predict



    def _predict(self, x_predict):

        distances = [self.distance_function(x_predict, X_row)for X_row in self.X]
        indexes = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y[i] for i in indexes]
        #return statistics.mode(k_nearest_labels) metodo usato in precedenza per svolgere il voto di maggioranza

        label_counts = Counter(k_nearest_labels)
        most_common = label_counts.most_common()
        max_votes = most_common[0][1]  # Il numero massimo di voti
        # Trova tutte le classi che hanno ricevuto max_votes (situazione di parità)
        candidates = [label for label, count in most_common if count == max_votes]

        if len(candidates) == 1:
            return candidates[0]
        else:
            # return min(candidates) #Soluzione che mi ritorna agli score di sklearn
            # return k_nearest_labels[0] # Mia soluzione iniziale
            k_min = 3
            candidate_label = min(candidates)
            label_encounter_dict = {label: 0 for label in set(k_nearest_labels)}
            for label in k_nearest_labels:
                label_encounter_dict[label] += 1
                if label_encounter_dict[label] > k_min:
                    candidate_label = label
                    break
            return candidate_label




    def plot_decision_regions(self, resolution=0.1):
        ## Effettuare SVD per ridurre il dataset a 2 features per poter plottare le decison region.
        if self.X.shape[1] > 2:
            U, sigma, V = truncated_svd(self.X, k=2)

            X_2D = self.X @ V
        else:
            X_2D = self.X

        self.fit(X_2D, self.y)

        ##2D soluzione semplice.
        x_min, x_max = X_2D[:, 0].min() - 1, X_2D[:, 0].max() + 1
        y_min, y_max = X_2D[:, 1].min() - 1, X_2D[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                             np.arange(y_min, y_max, resolution))

        grid_points = np.c_[xx.ravel(), yy.ravel()]

        # Disegna le regioni per metriche differenti
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))

        Z = self.predict(grid_points, metric='euclidean')
        Z = np.array(Z).reshape(xx.shape)
        axs[0, 0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        axs[0, 0].scatter(X_2D[:, 0], X_2D[:, 1], c=self.y, s=40, cmap='viridis', edgecolor='k', alpha=0.2)
        plt.subplots_adjust(top=0.95)
        axs[0, 0].set_title("Regioni di decisione | metric = euclidean ")

        Z = self.predict(grid_points, metric='manhattan')
        Z = np.array(Z).reshape(xx.shape)
        axs[0, 1].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        axs[0, 1].scatter(X_2D[:, 0], X_2D[:, 1], c=self.y, s=40, cmap='viridis', edgecolor='k', alpha=0.2)
        axs[0, 1].set_title("Regioni di decisione | metric = manhattan")

        Z = self.predict(grid_points, metric='cosine')
        Z = np.array(Z).reshape(xx.shape)
        axs[0, 2].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        axs[0, 2].scatter(X_2D[:, 0], X_2D[:, 1], c=self.y, s=40, cmap='viridis', edgecolor='k', alpha=0.2)
        axs[0, 2].set_title("Regioni di decisione | metric = cosine")

        Z = self.predict(grid_points, metric='chebyshev')
        Z = np.array(Z).reshape(xx.shape)
        axs[1, 0].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        axs[1, 0].scatter(X_2D[:, 0], X_2D[:, 1], c=self.y, s=40, cmap='viridis', edgecolor='k', alpha=0.2)
        axs[1, 0].set_title("Regioni di decisione | metric = chebyshev")

        Z = self.predict(grid_points, metric='minkowski', p=0.5)
        Z = np.array(Z).reshape(xx.shape)
        axs[1, 1].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        axs[1, 1].scatter(X_2D[:, 0], X_2D[:, 1], c=self.y, s=40, cmap='viridis', edgecolor='k', alpha=0.2)
        axs[1, 1].set_title("Regioni di decisione | metric = minkowski p = 0.5")

        Z = self.predict(grid_points, metric='minkowski', p=3)
        Z = np.array(Z).reshape(xx.shape)
        axs[1, 2].contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
        axs[1, 2].scatter(X_2D[:, 0], X_2D[:, 1], c=self.y, s=40, cmap='viridis', edgecolor='k', alpha=0.2)
        axs[1, 2].set_title("Regioni di decisione | metric = minkowski p=3")

        plt.show()
