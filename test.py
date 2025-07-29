import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs
import pandas as pd


dataset_multiclass, target_multiclass = make_blobs(n_samples=550, n_features=2, centers=5, random_state=42, cluster_std=3)
dataframe_multiclass = pd.DataFrame(dataset_multiclass)
dataframe_target_multiclass = pd.DataFrame(target_multiclass)
dataframe_multiclass["target"] = dataframe_target_multiclass.astype(str).agg("_".join, axis=1)


sns.pairplot(dataframe_multiclass, hue = "target", palette="viridis")
plt.show()

