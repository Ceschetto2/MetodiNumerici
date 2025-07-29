from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from KnnFromScratch import KnnFromScratch
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd

import numpy as np







def show_table(ax, df, title):
    ax.axis('off')
    tbl = ax.table(cellText=df.round(2).values,
                   colLabels=df.columns,
                   rowLabels=df.index,
                   loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1.2, 1.2)
    ax.set_title(title, fontsize=12)

def main():
    dataset, target= make_blobs(n_samples=550, n_features=4, centers=5, random_state=42,
                                                       cluster_std=5)
    sns.pairplot(pd.DataFrame(dataset).assign(target = target), hue="target", palette="viridis")
    plt.show()
    # centratura del datast
    scaler = StandardScaler()
    dataset_scaled = scaler.fit_transform(dataset)

    # split train/test
    x_train, x_test, y_train, y_test = train_test_split(dataset_scaled, target, test_size=0.2, random_state=42)

    ## Pairplot del train test
    sns.pairplot(pd.DataFrame(x_train).assign(target= y_train), hue='target', palette = "viridis")
    plt.suptitle("TrainSet", size=16)
    plt.subplots_adjust(top=0.95)
    plt.show()

    # Pairplot del test set
    sns.pairplot(pd.DataFrame(x_test).assign(target= y_test), hue='target', palette="viridis")
    plt.suptitle("TestSet", size=16)
    plt.subplots_adjust(top=0.95)
    plt.show()


    f1_scores_dict = {
        'euclidean': [],
        'manhattan': [],
        'cosine': [],
        'minkowski1': [], #mikowsky p = 1/2
        'minkowski2': [], #mikowsky p = 3
        'chebyshev': []
    }

    # lista do probabili vicini
    neighbor_values = list(range(1, 50, 2))

    k_optimal = 0
    max_f1 = 0
    best_metric = ''
    p = 2
    for n_neighbors in neighbor_values:

        kfs = KnnFromScratch(k=n_neighbors).fit(x_train, y_train)


        ### Confronto tra kfs con metriche differenti
        kfs_euclidean_prediction = kfs.predict(x_test, metric='minkowski', p=2)
        kfs_manhattan_prediction = kfs.predict(x_test, metric = 'minkowski', p = 1)
        kfs_chebyshev_prediction = kfs.predict(x_test, metric = 'chebyshev')
        kfs_minkowski1_prediction = kfs.predict(x_test, metric = 'minkowski', p=1/2)
        kfs_minkowski2_prediction = kfs.predict(x_test, metric = 'minkowski', p=3)
        kfs_cosine_prediction = kfs.predict(x_test, metric = 'cosine')



        # Genero i report
        kfs_euc_report_dict = classification_report(y_test, kfs_euclidean_prediction, output_dict=True)
        kfs_manh_report_dict = classification_report(y_test, kfs_manhattan_prediction, output_dict=True)
        kfs_cos_report_dict = classification_report(y_test, kfs_cosine_prediction, output_dict=True)
        kfs_cheby_report_dict = classification_report(y_test, kfs_chebyshev_prediction, output_dict=True)
        kfs_min1_report_dict = classification_report(y_test, kfs_minkowski1_prediction, output_dict=True)
        kfs_min2_report_dict = classification_report(y_test, kfs_minkowski2_prediction, output_dict=True)

        # aggiungo l'f1-score al dizionario per il plot successivamente
        f1_scores_dict['euclidean'].append(kfs_euc_report_dict['weighted avg']['f1-score'])
        f1_scores_dict['manhattan'].append(kfs_manh_report_dict['weighted avg']['f1-score'])
        f1_scores_dict['cosine'].append(kfs_cos_report_dict['weighted avg']['f1-score'])
        f1_scores_dict['chebyshev'].append(kfs_cheby_report_dict['weighted avg']['f1-score'])
        f1_scores_dict['minkowski1'].append(kfs_min1_report_dict['weighted avg']['f1-score'])
        f1_scores_dict['minkowski2'].append(kfs_min2_report_dict['weighted avg']['f1-score'])



        # aggiorno il massimo, il k e la metrica a cui è stato ottenuto
        max_f1_current = max(
            kfs_euc_report_dict['weighted avg']['f1-score'],
            kfs_manh_report_dict['weighted avg']['f1-score'],
            kfs_cos_report_dict['weighted avg']['f1-score'],
            kfs_cheby_report_dict['weighted avg']['f1-score'],
            kfs_min1_report_dict['weighted avg']['f1-score'],
            kfs_min2_report_dict['weighted avg']['f1-score'],

        )

        if max_f1_current > max_f1:

            if max_f1_current == kfs_euc_report_dict['weighted avg']['f1-score']:
                best_metric_current = 'euclidean'
            elif max_f1_current == kfs_manh_report_dict['weighted avg']['f1-score']:
                best_metric_current = 'manhattan'
            elif max_f1_current == kfs_cos_report_dict['weighted avg']['f1-score']:
                best_metric_current = 'cosine'
            elif max_f1_current == kfs_cheby_report_dict['weighted avg']['f1-score']:
                best_metric_current = 'chebyshev'
            elif max_f1_current == kfs_min1_report_dict['weighted avg']['f1-score']:
                best_metric_current = 'minkowski'
                p = 1/2
            else:
                best_metric_current = 'minkowski'
                p = 3


            max_f1 = max_f1_current
            k_optimal = n_neighbors
            best_metric = best_metric_current
            print(k_optimal, max_f1, best_metric, p)






    print(k_optimal, max_f1, best_metric, p)


    # Definizione dei colori e dei marker
    colors = {
        'euclidean': 'blue',
        'manhattan': 'green',
        'cosine': 'orange',
        'minkowski1': 'red',
        'minkowski2': 'gray',
        'chebyshev': 'purple',
    }

    markers = {
        'euclidean': 'o',  # Cerchio
        'manhattan': 's',  # Quadrato
        'cosine': '^',  # Triangolo
        'minkowski1': 'D',  # Diamante
        'minkowski2': 'x',  # Croce
        'chebyshev': '+',  # Più
    }

    # Plotting dei dati
    plt.figure(figsize=(12, 6))

    for metric, scores in f1_scores_dict.items():
        plt.plot(neighbor_values, scores, marker=markers[metric], label=metric.capitalize(), color=colors[metric])

    plt.title("F1-score vs Numero di vicini per diverse metriche (KNN From Scratch)")
    plt.xlabel("Numero di vicini (k)")
    plt.ylabel("F1-score (weighted avg)")

    plt.grid(True)
    plt.xticks(neighbor_values)
    plt.legend()
    plt.tight_layout()

    plt.show()




    ### Confronto tra KnnFromScratch, l'albero di decisione, la regressione logistica e il Knn di sklearn (con la stessa metrica e numero di vicini del miglior risultato ottenuto).
    kfs = KnnFromScratch(k=k_optimal).fit(x_train, y_train)
    dtc = DecisionTreeClassifier().fit(x_train, y_train)
    rfc = RandomForestClassifier(random_state=42).fit(x_train, y_train)
    knn = KNeighborsClassifier(n_neighbors=k_optimal, algorithm= 'brute', metric = best_metric, p = p).fit(x_train, y_train)


    kfs_prediction = kfs.predict(x_test, metric=best_metric, p = p)
    dtc_prediction = dtc.predict(x_test)
    knn_prediction = knn.predict(x_test)
    rfc_prediction = rfc.predict(x_test)

    # Ottieni i report come dizionari
    kfs_dict = classification_report(y_test, kfs_prediction, output_dict=True)
    dtc_dict = classification_report(y_test, dtc_prediction, output_dict=True)
    knn_dict = classification_report(y_test, knn_prediction, output_dict=True)
    rfc_dict = classification_report(y_test, rfc_prediction, output_dict=True)


    # Mostra i report
    fig, axes = plt.subplots(2, 2, figsize=(18, 6))
    show_table(axes[0,0], pd.DataFrame(kfs_dict).transpose(), f'KNN from Scratch | n_neighbors = {k_optimal} | distance = {best_metric}' )
    show_table(axes[0,1], pd.DataFrame(knn_dict).transpose(), f"Scikit-learn KNN | n_neighbors = {k_optimal} | distace = {best_metric}" )
    show_table(axes[1,0], pd.DataFrame(dtc_dict).transpose(), "Decision Tree Classifier")
    show_table(axes[1,1], pd.DataFrame(rfc_dict).transpose(), "Random Forest Classifier")
    plt.tight_layout()
    plt.show()

    ## Plot regioni di decisione per k_opt
    kfs.plot_decision_regions(resolution = 0.1)


if __name__ == '__main__':
    main()