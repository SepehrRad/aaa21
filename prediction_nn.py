import re
from fastai.tabular.all import *


def _preprocess_data_for_nn(df, temporal_resolution):
    """
    Doc String!
    """
    df = add_datepart(df, 'Trip Start Timestamp')
    df = df.astype({'Trip Start TimestampWeek': 'uint32'})
    df = df.astype({'Trip Start TimestampElapsed': 'int64'})
    target = f"Demand ({temporal_resolution})"
    # Only Columns with more than 12 distinct values will be consider as categories
    cont_vars, cat_vars = cont_cat_split(df, max_card=12)
    hex_regex = re.compile(".*hex")
    _ = list(filter(hex_regex.match, cont_vars))
    if bool(_):
        [cont_vars.remove(entry) for entry in _]
        cat_vars.extend(_)
    cont_vars.remove(target)
    return df, target, cont_vars, cat_vars


def split_data_sets_for_nn(df, temporal_resolution, test_size=0.2, validation_size=0.2, batch_size=32):
    """
    Doc String!
    """
    data, target, cont_vars, cat_vars = _preprocess_data_for_nn(df=df, temporal_resolution=temporal_resolution)
    train_index = int(len(data) * (1 - test_size))
    df_train = data[0:train_index]
    df_test = data[train_index:]

    # Setting indices for creating a validation set
    start_index = len(df_train) - int(len(df_train) * validation_size)
    end_index = len(df_train)

    # Test set
    test_data = TabularPandas(df_test, cat_names=cat_vars, cont_names=cont_vars, y_names=target,
                              procs=[Categorify, FillMissing, Normalize])
    val_index = df_train.iloc[start_index:end_index].index.values
    val_split = IndexSplitter(val_index)(range_of(df_train))
    # val_split = RandomSplitter(valid_pct=validation_size, seed=7)(range_of(df_train))

    # Create train validation data
    train_val_data = TabularPandas(df_train, procs=[Categorify, FillMissing, Normalize],
                                   cat_names=cat_vars,
                                   cont_names=cont_vars,
                                   y_names=target,
                                   batch_size=batch_size,
                                   y_block=RegressionBlock(n_out=1),
                                   splits=val_split)

    # Creating the final data loaders
    dls = train_val_data.dataloaders()

    # Test dl
    # test_dl = TabDataLoader(test_data, bs=batch_size, shuffle=False, drop_last=False)
    test_dl = dls.test_dl(df_test, with_labels=True)

    return dls, test_dl
