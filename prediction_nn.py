import re

from fastai.tabular.all import *
from prediction_utils import preprocess_data_for_prediction


def split_data_sets_for_nn(
    df, temporal_resolution, test_size=0.2, validation_size=0.2, batch_size=32
):
    """
    This function splits a data set into three data loaders, namely train,test, and validation
    ----------------------------------------------
    :param
        df (pandas.DataFrame): the given data set
        temporal_resolution (String): the target temporal resolution
        test_size (float): the test set size
        validation_size (float): the validation set size
        batch_size (int): int size of each mini batch
    :return
        DataLoaders : the train, validation and test data loaders
        DataLoader: the test data loader
        pandas.DataFrame: the test data frame
    """
    data, target, cont_vars, cat_vars = preprocess_data_for_prediction(df=df, temporal_resolution=temporal_resolution)
    train_index = int(len(data) * (1 - test_size))
    df_train = data[0:train_index]
    df_test = data[train_index:]

    # Setting indices for creating a validation set
    start_index = len(df_train) - int(len(df_train) * validation_size)
    end_index = len(df_train)

    val_index = df_train.iloc[start_index:end_index].index.values
    val_split = IndexSplitter(val_index)(range_of(df_train))

    # Create train validation data
    train_val_data = TabularPandas(
        df_train,
        procs=[Categorify, FillMissing, Normalize],
        cat_names=cat_vars,
        cont_names=cont_vars,
        y_names=target,
        y_block=RegressionBlock(n_out=1),
        splits=val_split,
    )

    # Creating the final data loaders
    dls = train_val_data.dataloaders(batch_size=batch_size)

    # Test dl
    test_dl = dls.test_dl(df_test, with_labels=True)

    return dls, test_dl, df_test
