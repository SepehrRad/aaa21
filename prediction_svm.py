from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV


def split_data_sets_for_svm(df, test_size=0.2):
    """
    This splits a given dataframe into a train and a test set.
    ----------------------------------------------
    :param
           df (pandas.DataFrame): The given data set.
           test_size (float): Defines the size of the test set. Default is 0.2.
    """
    train_index = int(len(df) * (1 - test_size))
    df_train = df[0:train_index]
    df_test = df[train_index:]
    return df_train, df_test


def make_pipeline_for_svm(cat_vars, cont_vars, model):
    """
    This builds a support vector regression pipeline based on a given list of categorical and continuous variables
    and a model.
    ----------------------------------------------
    :param
           cat_vars (list): List of categorical variables.
           cont_vars (list): List of continuous variables.
           model: Used model for prediction.
    """
    numeric_transformer = Pipeline(steps=[('standard_scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('one_hot_encoder', OneHotEncoder(handle_unknown='ignore'))])
    svr_preprocessor = ColumnTransformer(
        transformers=[
            ('numerical scaler', numeric_transformer, cont_vars),
            ('one hot encoder', categorical_transformer, cat_vars)
        ]
    )
    svr_model = model
    svr_pipeline = Pipeline(steps=[
        ('preprocessor', svr_preprocessor),
        ('svr model', svr_model)
    ])
    return svr_pipeline


def find_best_parameters_for_model(
        pipeline, X_train, y_train, model_params, scoring, n_iter, verbose=True):
    """
    This function performs a randomized grid search with five time series splits on the training set.
    ----------------------------------------------
    :param
           pipeline (sklearn.pipeline): The pipeline with the model and transformers which will be used for grid search.
           X_train (pandas.DataFrame): Training features.
           y_train (pandas.DataFrame): Target.
           model_params (dictionary): Used model parameters for the grid search.
           scoring (String): The scoring metric used for grid search.
           n_iter (int): The number of performed grid searches.
           verbose (boolean): Print detailed information while performing grid search.
    """
    print(f"Running grid search for the model based on {scoring}")
    grid_pipeline = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=model_params,
        n_jobs=-1,
        n_iter=n_iter,
        cv=TimeSeriesSplit(n_splits=5),
        scoring=scoring,
        random_state=42,
        verbose=verbose,
    )
    grid_pipeline.fit(X_train, y_train)
    print(f"Best {scoring} Score was: {grid_pipeline.best_score_}")
    print(f"The best hyper parameters for the model are:")
    print(grid_pipeline.best_params_)
