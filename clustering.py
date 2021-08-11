import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import math




def get_silhouette_score(df, n_clusters, n_init=20, init_params='kmeans',
                         metric="euclidean", sample_size=1000, random_state=7, save_plot=False, save_name="silhouette"):
    s_score = []
    clusters = range(2, n_clusters + 2)

    for cluster in clusters:
        model = GaussianMixture(n_components=cluster, n_init=n_init, init_params=init_params)
        labels = model.fit_predict(df)
        s_score.append(silhouette_score(df, labels, metric=metric, sample_size=sample_size, random_state=random_state))

    # Plot the resulting Silhouette scores on a graph
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x=clusters, y= s_score, palette="dark", marker="o")
        ax.set_title('Identify the Number of Clusters using Silhouette Score', fontsize=16, fontweight='bold', pad=20)
        ax.set(xlabel='Cluster', ylabel='BIC')
        fig.tight_layout()

    if save_plot:
        ax.figure.savefig(f'img/{save_name}.png', bbox_inches='tight', dpi=1000)


def get_elbow(df, n_clusters, save_plot=False, save_name="elbow"):
    inertia = []
    K = range(1, n_clusters + 2)

    for k in K:
        kmeansModel = KMeans(n_clusters=k, init='k-means++', random_state=7)
        kmeansModel.fit(df)
        inertia.append(kmeansModel.inertia_)
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x=K, y=inertia, palette="dark", marker="o")
        ax.set_title('Elbow Method Showing the Optimal Number of Clusters', fontsize=16, fontweight='bold', pad=20)
        ax.set(xlabel='Cluster', ylabel='Inertia')
        fig.tight_layout()

    if save_plot:
        ax.figure.savefig(f'img/{save_name}.png', bbox_inches='tight', dpi=1000)

def get_bic(df, n_clusters, n_init=20, init_params='kmeans', save_plot=False, save_name="bic"):

    bic_score = []
    clusters = range(2, n_clusters + 2)

    for cluster in clusters:
        model = GaussianMixture(n_components=cluster, n_init=n_init, init_params=init_params).fit(df)
        bic_score.append(model.bic(df))
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x=clusters, y=bic_score, palette="dark", marker="o")
        ax.set_title('Identify the Number of Clusters using BIC', fontsize=16, fontweight='bold', pad=20)
        ax.set_xticks(clusters)
        ax.set(xlabel='Cluster', ylabel='BIC')
        fig.tight_layout()

    if save_plot:
        ax.figure.savefig(f'img/{save_name}.png', bbox_inches='tight', dpi=1000)

def _get_clusters_sizes(cluster, save_plot=False, save_name="sizes"):
    (unique, counts) = np.unique(cluster, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    frequencies = pd.DataFrame(data=frequencies, columns=["Cluster", "Count"])

    fig = plt.figure(figsize=(12, 12))
    ax = sns.barplot(data=frequencies, x="Cluster", y="Count", palette="dark")
    ax.set_title("Cluster Size", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    fig.tight_layout()

    if save_plot:
        ax.figure.savefig(f'img/{save_name}.png', bbox_inches='tight', dpi=1000)


def _plot_clusters_boxplot(X, column_1, column_2, save_plot=False, save_name="box"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.boxplot(x="cluster", y=column_1, ax=ax1, data=X, palette="dark")
    ax1.set_title(f'Cluster - Feature {column_1}', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xlabel('Cluster')
    ax1.set_ylabel(column_1)
    sns.boxplot(x="cluster", y=column_2, ax=ax2, data=X, palette="dark")
    ax2.set_title(f'Cluster - Feature {column_2}', fontsize=16, fontweight='bold', pad=20)
    ax2.set_xlabel('Cluster')
    ax2.set_ylabel(column_2)
    fig.tight_layout()

    if save_plot:
        ax1.figure.savefig(f'img/{save_name}_1.png', bbox_inches='tight', dpi=1000)
        ax2.figure.savefig(f'img/{save_name}_2.png', bbox_inches='tight', dpi=1000)


def _plot_clusters_scatterplot(X, column_1, column_2, title, xlabel, ylabel,save_plot=False, save_name="scatter"):
    fig = plt.figure(figsize=(9, 9))
    ax = sns.scatterplot(data=X, x=column_1, y=column_2, hue='cluster', s=20, alpha=0.1, palette="dark")
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Cluster')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

    if save_plot:
        ax.figure.savefig(f'img/{save_name}.png', bbox_inches='tight', dpi=1000)

def _plot_clusters_parallel_coordinates(df, title):

    X = df.groupby("cluster").sample(n=50, random_state=7)

    fig = px.parallel_coordinates(X,
                                  dimensions=X[X.columns.difference(["cluster"])],
                                  color="cluster",
                                  title=title,
                                  )

    fig.show()



def _plot_clusters_pairplot(X, title, save_plot, save_name):
    fig = plt.figure(figsize=(20, 20))
    ax = sns.pairplot(X,
                      hue="cluster",
                      corner=True,
                      palette="dark",
                      height=3.5,
                      aspect=2,
                      )
    ax.fig.suptitle(title, y=1.08)
    fig.tight_layout()
    if save_plot:
        ax.savefig(f'img/{save_name}.png', bbox_inches='tight', dpi=1000)


def _cyclical_feature_transformer(cyclical_col):
    """
    This function maps cyclical features to two distinct components (sine & cosine) & return them as vector.
    ----------------------------------------------
    :param
        cyclical_col (numpy.ndarray): cyclical feature vector
    :returns
        numpy.ndarray: sine & cosine vectors for the given feature vector
        {
        sine,
        cosine
        }
    """
    maximum_value = np.amax(cyclical_col)
    sin_comp = np.sin((2 * math.pi * cyclical_col) / maximum_value)
    cos_comp = np.cos((2 * math.pi * cyclical_col) / maximum_value)
    return sin_comp, cos_comp


def _categorical_feature_transformer(df, categorical_col_names, drop_first=False):
    """
    This function encodes the categorical variables in a given data frame using one hot encoding method.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): the given pandas data frame with categorical features
        categorical_col_names (string[]): the name of the categorical features as a list
        drop_first (bool): the decision to drop one category per feature
    :returns
        pandas.DataFrame: dummy-coded DataFrame
    """
    df[categorical_col_names] = df[categorical_col_names].astype("str")
    return pd.get_dummies(df[categorical_col_names], drop_first=drop_first)


def _numerical_feature_transformer(df, numerical_col_names):
    """

    ----------------------------------------------
    :param
        df (pandas.DataFrame): the given pandas data frame with numerical features
        categorical_col_names (string[]): the name of the categorical features as a list
        drop_first (bool): the decision to drop one category per feature
    :returns
        pandas.DataFrame: dummy-coded DataFrame
    """
    scaler = StandardScaler()
    df[numerical_col_names] = scaler.fit_transform(df[numerical_col_names])
    return df


def transform_columns(df, col_dict, drop_cols=True, drop_first=False):
    """
    This function transforms the cyclical & categorical columns of any given data frame.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): the given pandas data frame with categorical & cyclical features
        col_dict (dict): the name of the categorical & cyclical features
        drop_first (bool): the decision to drop one category for categorical features
        drop_cols(bool): the decision to remove the original features from the given data frame after transformation
    :returns
        pandas.DataFrame: the data frame with transformed features
    :raises
        ValueError: if col_dict or df are empty/null
    """

    if col_dict is None or len(col_dict) == 0:
        raise ValueError("The columns dictionary can not be null!")
    if df is None:
        raise ValueError("The data frame can not be null!")

    cyclical_features = col_dict.get("cyclical_features")
    if len(cyclical_features) != 0:
        for feature in cyclical_features:
            (
                df[f"{feature}_sine"],
                df[f"{feature}_cosine"],
            ) = _cyclical_feature_transformer(df[feature])

    categorical_features = col_dict.get("categorical_features")
    if len(categorical_features) != 0:
        _ = _categorical_feature_transformer(
            df, categorical_features, drop_first=drop_first
        )
        df = df.join(_)

    numerical_features = col_dict.get("numerical_features")
    if len(numerical_features) != 0:
        df = _numerical_feature_transformer(df, numerical_features)

    if drop_cols:
        df.drop(cyclical_features, inplace=True, axis=1)
        df.drop(categorical_features, inplace=True, axis=1)

    return df


def get_clusters_gmm(X, title, xlabel, ylabel, n_cluster, n_init=20, init_params='kmeans',
                     random_state=7, plot_sizes=False, plot_boxes=False, save_plots=False):
    gmm = GaussianMixture(n_components=n_cluster, random_state=random_state, n_init=n_init,
                          init_params=init_params).fit(X)
    cluster = gmm.predict(X)
    cluster_proba = gmm.predict_proba(X)
    X['cluster'] = cluster

    if len(X.columns) == 4:
        _plot_clusters_scatterplot(X,
                                   X.columns[0],
                                   X.columns[1],
                                   title,
                                   xlabel,
                                   ylabel,
                                   save_plot=save_plots,
                                   save_name=f"scatter_gmm_{X.columns[0]}_{X.columns[1]}")

        if plot_boxes is True:
            _plot_clusters_boxplot(X,
                                   X.columns[0],
                                   X.columns[1],
                                   save_plot=save_plots,
                                   save_name=f"boxplot_gmm_{X.columns[0]}_{X.columns[1]}")

    else:
        _plot_clusters_parallel_coordinates(X, title)

    _plot_clusters_pairplot(X, title,
                            save_plot=save_plots,
                            save_name=f"pairplot_gmm_{X.columns[0]}_{X.columns[-1]}")

    if plot_sizes is True:
        _get_clusters_sizes(cluster, save_plot=save_plots,
                            save_name=f"sizes_gmm_{X.columns[0]}_{X.columns[-1]}")

    for k in range(n_cluster):
        X[f'cluster_{k}_prob'] = cluster_proba[:, k]
    return X


def get_clusters_kmeans(X, title, xlabel, ylabel, n_cluster, random_state=7,
                        plot_sizes=False, plot_boxes=False, save_plots=False):
    kmm = KMeans(n_clusters=n_cluster, init='k-means++', random_state=random_state).fit(X)
    cluster = kmm.predict(X)
    X['cluster'] = cluster

    if len(X.columns) == 3:
        _plot_clusters_scatterplot(X, X.columns[0], X.columns[1], title, xlabel, ylabel,
                                   save_plot=save_plots,
                                   save_name=f"scatter_kmeans_{X.columns[0]}_{X.columns[2]}")

        if plot_boxes is True:
            _plot_clusters_boxplot(X, X.columns[0], X.columns[1],
                                   save_plot=save_plots,
                                   save_name=f"boxplot_kmeans_{X.columns[0]}_{X.columns[1]}")

    else:
        _plot_clusters_parallel_coordinates(X, title)

    _plot_clusters_pairplot(X, title,
                            save_plot=save_plots,
                            save_name=f"pairplot_kmeans_{X.columns[0]}_{X.columns[-1]}")

    if plot_sizes is True:
        _get_clusters_sizes(cluster,
                            save_plot=save_plots,
                            save_name=f"sizes_kmeans_{X.columns[0]}_{X.columns[-1]}")

    return X
