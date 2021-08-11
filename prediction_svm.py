from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def split_data_sets_for_svm(df, test_size=0.2):
    """
    Dos String!
    """
    train_index = int(len(df) * (1-test_size))
    df_train = df[0:train_index]
    df_test = df[train_index:]
    return df_train, df_test


def make_pipeline_for_svm(cat_vars, cont_vars, model):
    """
    Doc String!
    """
    numeric_transformer = Pipeline(steps=[('standard_scaler', StandardScaler())])
    categorical_tranformer = Pipeline(steps=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))])
    svr_preprocessor = ColumnTransformer(
        transformers=[
            ('numerical scaler', numeric_transformer, cont_vars),
            ('one hot encoder', categorical_tranformer, cat_vars)
        ]
    )
    svr_model = model
    svr_pipeline = Pipeline(steps=[
        ('preprocessor', svr_preprocessor),
        ('svr model', svr_model)
    ])
    return svr_pipeline
