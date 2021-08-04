import re
from fastai.tabular.all import *


def _preprocess_data_for_nn(df, temporal_resolution):
    """
    Doc String!
    """

    if temporal_resolution == 'D':
        df = add_datepart(df, 'Trip Start Timestamp', prefix='')
    else:
        df = add_datepart(df, 'Trip Start Timestamp', prefix='', time=True)
        _ = [c for c in df.columns if ('minute' in c.lower() or 'second' in c.lower())]
        df.drop(_, axis=1, inplace=True)
    df = df.astype({'Week': 'uint32'})
    df = df.astype({'Elapsed': 'int64'})
    # Only Columns with more than 35 distinct values will be considered as continuous
    target = f"Demand ({temporal_resolution})"
    cont_vars, cat_vars = cont_cat_split(df, max_card=35, dep_var=target)
    # Spacial temporal columns that should be considered as categories in an embedding structure
    cont_vars.remove('Dayofyear')
    cat_vars.append('Dayofyear')
    cont_vars.remove('Week')
    cat_vars.append('Week')
    hex_regex = re.compile(".*hex")
    _ = list(filter(hex_regex.match, cont_vars))
    if bool(_):
        [cont_vars.remove(entry) for entry in _]
        cat_vars.extend(_)
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
                                   y_block=RegressionBlock(n_out=1),
                                   splits=val_split)

    # Creating the final data loaders
    dls = train_val_data.dataloaders(batch_size=batch_size)

    # Test dl
    # test_dl = TabDataLoader(test_data, bs=batch_size, shuffle=False, drop_last=False)
    test_dl = dls.test_dl(df_test, with_labels=True)

    return dls, test_dl
