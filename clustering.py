import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns


def get_silhouette_score(df, column_1, column_2,  n_clusters,  n_init=20, init_params='kmeans',
                         metric="euclidean", sample_size=1000, random_state=7):

    silhouette_score = []
    df = df[[column_1, column_2]]

    clusters = range(2, n_clusters+2)

    for cluster in clusters:
        model = GaussianMixture(n_components=cluster, n_init=n_init, init_params=init_params)
        labels = model.fit_predict(df)
        silhouette_score.append(silhouette_score(df, labels, metric=metric, sample_size=sample_size, random_state=random_state))

    # Plot the resulting Silhouette scores on a graph
    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(clusters, silhouette_score, 'bo-', color='black')
    plt.xlabel('n_clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()


def get_ellbow(df, range):
    distortions = []
    K = range(1, range+2)

    for k in K:
        kmeansModel = KMeans(n_clusters=k, init='k-means++', random_state=7)
        kmeansModel.fit(df)
        distortions.append(kmeansModel.inertia_)

    fig = plt.figure(figsize=(6, 6))
    ax = sns.lineplot(x=K, y=distortions_chicago, color='C3')
    ax.set_title('Elbow method showing the optimal k', fontsize=16, fontweight='bold', pad=20)
    ax.set(xlabel='K', ylabel='Inertia')
    fig.tight_layout()


def _get_clusters_sizes(cluster):
    (unique, counts) = np.unique(cluster, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    frequencies = pd.DataFrame(data=frequencies, columns=["Cluster", "Count"])

    fig = plt.figure(figsize=(5, 5))
    ax = sns.barplot(data=frequencies, x="Cluster", y="Count", palette='bright')
    ax.set_title("Cluster Size", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    fig.tight_layout()

def _plot_clusters_boxplot(X, column_1, column_2,):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    sns.boxplot(x="cluster", y=column_1, ax=ax1, data=X,
                palette=['green', 'blue', 'darkorchid', 'crimson', 'orange', 'darkturquoise'])
    ax1.set_title(f'Cluster - Feature {column_1}', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel(column_1)
    sns.boxplot(x="cluster", y=column_2, ax=ax2, data=X,
                palette=['green', 'blue', 'darkorchid', 'crimson', 'orange', 'darkturquoise'])
    ax2.set_title(f'Cluster - Feature {column_2}', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel(column_2)
    fig.tight_layout()

def get_clusters_gmm(df, column_1, column_2, title, xlabel, ylabel, n_cluster, n_init=20, init_params='kmeans', random_state=7, plot_sizes=False, plot_boxes = False):

    X = df[[column_1, column_2]]
    gmm = GaussianMixture(n_components=n_cluster, random_state=random_state, n_init=n_init, init_params=init_params).fit(X)
    cluster = gmm.predict(X)
    cluster_proba = gmm.predict_proba(X)
    X['cluster'] = cluster
    for k in range(n_cluster):
        X[f'cluster_{k}_prob'] = cluster_proba[:, k]
    fig = plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(data=X, x=column_1, y=column_2, hue='cluster', s=10, alpha=0.1,
                         palette='bright')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Cluster')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if plot_sizes is True:
        _get_cluster_sizes(cluster)

    if plot_boxes is True:
        _plot_cluster_boxplot(X, column_1, column_2)
    return X, gmm


def get_clusters_kmeans(df,column_1, column_2, numerical_columns, title, xlabel, ylabel, n_cluster, random_state=7, plot_sizes=False, plot_boxes = False):

    X = df[[column_1, column_2]]

    # scale numerical columns
    ct = ColumnTransformer([('Standard Scaler', StandardScaler(), numerical_columns)])
    X[numerical_columns] = ct.fit_transform(X)

    kmm = KMeans(n_clusters = n_cluster, init='k-means++', random_state=random_state).fit(X)
    cluster = kmm.predict(X)
    X['cluster'] = cluster

    fig = plt.figure(figsize=(5, 5))
    ax = sns.scatterplot(data=X, x=column_1, y=column_2, hue='cluster', s=10, alpha=0.1,
                         palette='bright')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Cluster')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if plot_sizes is True:
        _get_cluster_sizes(cluster)

    if plot_boxes is True:
        _plot_cluster_boxplot(X, column_1, column_2)
    return X, kmm

