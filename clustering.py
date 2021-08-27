import math

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


def get_silhouette_score(
    df,
    n_clusters,
    n_init=20,
    init_params="kmeans",
    metric="euclidean",
    sample_size=1000,
    random_state=7,
    save_plot=False,
    save_name="silhouette",
):
    """
    This function calculates the silhouette score for n clusters with GMMand plots the result as line chart.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): The given pandas data frame with features to use for clustering.
        n_clusters (int): Number of clusters.
        n_init (int): The number of initializations to perform. The best results are kept.
        init_params (string): Method used to initialize the weights, the means and the precision. Must be "kmeans" or
        "random".
        metric (string): Metirc to calculate the distance between instances in a feature array.
        sample_size (int): Sample size to use for computing the Silhouette Coefficent on a random subset.
        random_state (int): Random state to use.
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """

    s_score = []
    clusters = range(2, n_clusters + 2)

    for cluster in clusters:
        model = GaussianMixture(
            n_components=cluster, n_init=n_init, init_params=init_params
        )
        labels = model.fit_predict(df)
        s_score.append(
            silhouette_score(
                df,
                labels,
                metric=metric,
                sample_size=sample_size,
                random_state=random_state,
            )
        )

    # Plot the resulting Silhouette scores on a graph
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x=clusters, y=s_score, palette="bright", marker="o")
        ax.set_title(
            "Identify the Number of Clusters using Silhouette Score",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set(xlabel="Cluster", ylabel="Silhouette Score")
        fig.tight_layout()

    if save_plot:
        fig.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


def get_elbow(df, n_clusters, save_plot=False, save_name="elbow"):
    """
    This function calculates the distortion score for n_cluster and plots the result as line chart (
    also known as elbow method).
    ----------------------------------------------
    :param
        df (pandas.DataFrame): The given pandas data frame with features to use for clustering.
        n_clusters (int): Number of clusters.
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """
    inertia = []
    K = range(1, n_clusters + 2)

    for k in K:
        kmeansModel = KMeans(n_clusters=k, init="k-means++", random_state=7)
        kmeansModel.fit(df)
        inertia.append(kmeansModel.inertia_)
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x=K, y=inertia, palette="bright", marker="o")
        ax.set_title(
            "Elbow Method Showing the Optimal Number of Clusters",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set(xlabel="Cluster", ylabel="Inertia")
        fig.tight_layout()

    if save_plot:
        fig.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


def get_bic(
    df, n_clusters, n_init=20, init_params="kmeans", save_plot=False, save_name="bic"
):
    """
    This function calculates the BIC for n clusters with GMM and plots the result as line chart.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): The given pandas data frame with features to use for clustering.
        n_clusters (int): Number of clusters.
        n_init (int): The number of initializations to perform. The best results are kept.
        init_params (string): Method used to initialize the weights, the means and the precision. Must be "kmeans" or
        "random".
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """
    bic_score = []
    clusters = range(2, n_clusters + 2)

    for cluster in clusters:
        model = GaussianMixture(
            n_components=cluster, n_init=n_init, init_params=init_params
        ).fit(df)
        bic_score.append(model.bic(df))
    with sns.axes_style("darkgrid"):
        fig = plt.figure(figsize=(16, 8))
        ax = sns.lineplot(x=clusters, y=bic_score, palette="bright", marker="o")
        ax.set_title(
            "Identify the Number of Clusters using BIC",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(clusters)
        ax.set(xlabel="Cluster", ylabel="BIC")
        fig.tight_layout()

    if save_plot:
        fig.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


def plot_clusters_sizes(X, title, save_plot=False, save_name="sizes"):
    """
    This function plots the size of each cluster in a bar plot.
    ----------------------------------------------
    :param
        X (pandas.DataFrame): The given pandas data frame including a column "cluster".
        title (string): Title of the plot.
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """
    fig = plt.figure(figsize=(12, 12))
    ax = sns.countplot(x="cluster", data=X, palette="bright")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    fig.tight_layout()

    if save_plot:
        fig.figure.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


def plot_clusters_boxplot(X, column_1, save_plot=False, save_name="box"):
    """
    This function plots the distribution of two features in regards to the clusters as box plots.
    ----------------------------------------------
    :param
        X (pandas.DataFrame): The given pandas data frame including a column "cluster".
        column_1: First feature of DataFrame X to be used for the plot.
        column_2: Second feature of DataFrame X to be used for the plot.
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """

    fig = plt.figure(figsize=(9, 9))
    ax = sns.boxplot(x="cluster", y=column_1, data=X, palette="bright")
    ax.set_title(
        f"Cluster - Feature {column_1}", fontsize=16, fontweight="bold", pad=20
    )
    ax.set_xlabel("Cluster")
    ax.set_ylabel(column_1)
    fig.tight_layout()

    if save_plot:
        fig.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


def plot_clusters_scatter(
    X, column_1, column_2, title, save_plot=False, save_name="scatter"
):
    """
    This function plots two features in regards to the clusters as scatterplot.
    ----------------------------------------------
    :param
        X (pandas.DataFrame): The given pandas data frame including a column "cluster".
        column_1: First feature of DataFrame X to be used for the plot as x-axis.
        column_2: Second feature of DataFrame X to be used for the plot as y-axis.
        title (string): Title of the plot.
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """
    fig = plt.figure(figsize=(9, 9))
    ax = sns.scatterplot(
        data=X, x=column_1, y=column_2, hue="cluster", s=20, alpha=0.1, palette="bright"
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0, title="Cluster")
    ax.set_title(title, fontsize=16, fontweight="bold", pad=20)
    ax.set_xlabel(column_1)
    ax.set_ylabel(column_2)
    fig.tight_layout()

    if save_plot:
        fig.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


def plot_clusters_pairplot(X, title, save_plot=False, save_name="pair"):
    """
    This function plots a pair plot of all features in a DataFrame.
    ----------------------------------------------
    :param
        X (pandas.DataFrame): The given pandas data frame including a column "cluster".
        title (string): Title of the plot.
        save_plot (bool): Saves the plot as png in wd/img.
        save_name (string): Name of file.
    :returns
        None
    """
    fig = plt.figure(figsize=(20, 20))
    ax = sns.pairplot(
        X,
        hue="cluster",
        corner=True,
        palette="bright",
        height=3.5,
        aspect=2,
    )
    ax.fig.suptitle(title, y=1.08)
    fig.tight_layout()
    if save_plot:
        fig.savefig(f"img/{save_name}.png", bbox_inches="tight", dpi=1000)


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


def get_clusters_gmm(X, n_cluster, n_init=20, init_params="kmeans", random_state=7):
    """
    This function creates a Gaussian Mixture Model and returns a DataFrame with a prediction for the cluster of each
    data sample.
    ----------------------------------------------
    :param
        X (pandas.DataFrame): The given pandas data frame with features to use for clustering.
        n_clusters (int): Number of clusters.
        n_init (int): The number of initializations to perform. The best results are kept.
        init_params (string): Method used to initialize the weights, the means and the precision. Must be "kmeans" or
        "random".
        random_state (int): Random state to use.
    :returns
        pandas.DataFrame: DataFrame with information with a prediction for the cluster of each data sample.
    """

    gmm = GaussianMixture(
        n_components=n_cluster,
        random_state=random_state,
        n_init=n_init,
        init_params=init_params,
    ).fit(X)
    cluster = gmm.predict(X)
    cluster_proba = gmm.predict_proba(X)
    X["cluster"] = cluster
    for k in range(n_cluster):
        X[f"cluster_{k}_prob"] = cluster_proba[:, k]
    return X


def get_clusters_kmeans(X, n_cluster, random_state=7):
    """
    This function creates a kmeans++ and returns a DataFrame with a prediction for the cluster of each
    data sample.
    ----------------------------------------------
    :param
        X (pandas.DataFrame): The given pandas data frame with features to use for clustering.
        n_clusters (int): Number of clusters.
        random_state (int): Random state to use.
    :returns
        pandas.DataFrame: DataFrame with information with a prediction for the cluster of each data sample.
    """
    kmm = KMeans(n_clusters=n_cluster, init="k-means++", random_state=random_state).fit(
        X
    )
    cluster = kmm.predict(X)
    X["cluster"] = cluster
    return X
