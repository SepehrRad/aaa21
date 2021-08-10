import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import seaborn as sns
from pandas.plotting import parallel_coordinates


def get_silhouette_score(df, column_1, column_2, n_clusters, n_init=20, init_params='kmeans',
                         metric="euclidean", sample_size=1000, random_state=7):
    s_score = []
    df = df[[column_1, column_2]]

    clusters = range(2, n_clusters + 2)

    for cluster in clusters:
        model = GaussianMixture(n_components=cluster, n_init=n_init, init_params=init_params)
        labels = model.fit_predict(df)
        s_score.append(silhouette_score(df, labels, metric=metric, sample_size=sample_size, random_state=random_state))

    # Plot the resulting Silhouette scores on a graph
    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(clusters, s_score, 'bo-', color='black')
    plt.xlabel('n_clusters')
    plt.ylabel('Silhouette Score')
    plt.title('Identify the number of clusters using Silhouette Score')
    plt.show()


def get_ellbow(df, n_clusters):
    distortions = []
    K = range(1, n_clusters + 2)

    for k in K:
        kmeansModel = KMeans(n_clusters=k, init='k-means++', random_state=7)
        kmeansModel.fit(df)
        distortions.append(kmeansModel.inertia_)

    fig = plt.figure(figsize=(16, 8))
    ax = sns.lineplot(x=K, y=distortions, color='C3')
    ax.set_title('Elbow method showing the optimal k', fontsize=16, fontweight='bold', pad=20)
    ax.set(xlabel='K', ylabel='Inertia')
    fig.tight_layout()


def get_bic(df, column_1, column_2, n_clusters, n_init=20, init_params='kmeans'):
    bic_score = []
    df = df[[column_1, column_2]]

    clusters = range(2, n_clusters + 2)

    for cluster in clusters:
        model = GaussianMixture(n_components=cluster, n_init=n_init, init_params=init_params).fit(df)
        bic_score.append(model.bic(df))

    # Plot the resulting Silhouette scores on a graph
    plt.figure(figsize=(16, 8), dpi=300)
    plt.plot(clusters, bic_score, 'bo-', color='blue')
    plt.xlabel('n_clusters')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Identify the Number of Clusters using AIC and BIC')
    plt.show()


def _get_clusters_sizes(cluster):
    (unique, counts) = np.unique(cluster, return_counts=True)
    frequencies = np.asarray((unique, counts)).T
    frequencies = pd.DataFrame(data=frequencies, columns=["Cluster", "Count"])

    fig = plt.figure(figsize=(12, 12))
    ax = sns.barplot(data=frequencies, x="Cluster", y="Count",
                     palette=['green', 'blue', 'darkorchid', 'crimson', 'orange', 'darkturquoise'])
    ax.set_title("Cluster Size", fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Size")
    fig.tight_layout()


def _plot_clusters_boxplot(X, column_1, column_2, ):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
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


def _plot_clusters_scatterplot(X, column_1, column_2, title, xlabel, ylabel, ):
    fig = plt.figure(figsize=(9, 9))
    ax = sns.scatterplot(data=X, x=column_1, y=column_2, hue='cluster', s=20, alpha=0.1,
                         palette=['green', 'blue', 'darkorchid', 'crimson', 'orange', 'darkturquoise'])
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Cluster')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()

def _plot_clusters_parallel_coordinates(X, title, xlabel, ylabel):
    fig = plt.figure(figsize=(9, 9))
    ax = parallel_coordinates(frame=X,
                              class_column=X["cluster"],
                              color=['green', 'blue', 'darkorchid', 'crimson', 'orange', 'darkturquoise'],
                              )
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title='Cluster')
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    fig.tight_layout()


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

    if drop_cols:
        df.drop(cyclical_features, inplace=True, axis=1)
        df.drop(categorical_features, inplace=True, axis=1)

    return df


def get_clusters_gmm(df, column_1, column_2, title, xlabel, ylabel, n_cluster, n_init=20, init_params='kmeans',
                     random_state=7, plot_sizes=False, plot_boxes=False):
    X = df[[column_1, column_2]]
    gmm = GaussianMixture(n_components=n_cluster, random_state=random_state, n_init=n_init,
                          init_params=init_params).fit(X)
    cluster = gmm.predict(X)
    cluster_proba = gmm.predict_proba(X)
    X['cluster'] = cluster
    for k in range(n_cluster):
        X[f'cluster_{k}_prob'] = cluster_proba[:, k]

    _plot_clusters_scatterplot(X, column_1, column_2, title, xlabel, ylabel)

    if plot_sizes is True:
        _get_clusters_sizes(cluster)

    if plot_boxes is True:
        _plot_clusters_boxplot(X, column_1, column_2)
    # return X, gmm


def get_clusters_kmeans(df, column_1, column_2, numerical_columns, title, xlabel, ylabel, n_cluster, random_state=7,
                        plot_sizes=False, plot_boxes=False):
    X = df[[column_1, column_2]]

    # scale numerical columns
    # ct = ColumnTransformer([('Standard Scaler', StandardScaler(), numerical_columns)])
    # X[numerical_columns] = ct.fit_transform(X)

    kmm = KMeans(n_clusters=n_cluster, init='k-means++', random_state=random_state).fit(X)
    cluster = kmm.predict(X)
    X['cluster'] = cluster

    _plot_clusters_scatterplot(X, column_1, column_2, title, xlabel, ylabel)

    if plot_sizes is True:
        _get_clusters_sizes(cluster)

    if plot_boxes is True:
        _plot_clusters_boxplot(X, column_1, column_2)
    # return X, kmm

def get_clusters_kmeans_md(X, title, xlabel, ylabel, n_cluster, random_state=7, plot_sizes=False, plot_boxes=False):
    # scale numerical columns
    # ct = ColumnTransformer([('Standard Scaler', StandardScaler(), numerical_columns)])
    # X[numerical_columns] = ct.fit_transform(X)

    kmm = KMeans(n_clusters=n_cluster, init='k-means++', random_state=random_state).fit(X)
    cluster = kmm.predict(X)
    X['cluster'] = cluster

    #_plot_clusters_scatterplot(X, column_1, X, title, xlabel, ylabel)
    _plot_clusters_parallel_coordinates(X, title, xlabel, ylabel)
    if plot_sizes is True:
        _get_clusters_sizes(cluster)

    if plot_boxes is True:
        _plot_clusters_boxplot(X, column_1, X)
    # return X, kmm
