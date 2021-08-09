from sklearn import metrics
import math
import numpy as np


def _smape(y_true, y_predicted):
    """
    This function calculates the sMAPE score.
    ----------------------------------------------
    :param
        y_true (List): Test values
        y_true (List): Predicted values
    :return
        float : sMAPE Score
    """
    y_true = np.array([i for i in y_true])
    y_predicted = np.array([i for i in y_predicted])
    sMape = (100 / len(y_true) * np.sum(2 * np.abs(y_predicted - y_true) /
                                        (np.abs(y_true) + np.abs(y_predicted))))[0]
    return float(sMape)


def get_prediction_scores(y_true, y_predicted, s_mape=False):
    """
    This function prints the prediction scores.
    ----------------------------------------------
    :param
        y_true (List): Test values
        y_true (List): Predicted values
    """
    print("-------MODEL SCORES-------")
    print(f"MAPE: {100 * metrics.mean_absolute_percentage_error(y_true, y_predicted): .3f} %")
    if s_mape: print(f"sMAPE: {_smape(y_true, y_predicted): .3f} %")
    print(f"MAE: {metrics.mean_absolute_error(y_true, y_predicted): .3f}")
    print(f"MSE: {metrics.mean_squared_error(y_true, y_predicted): .3f}")
    print(f"RMSE: {math.sqrt(metrics.mean_squared_error(y_true, y_predicted)): .3f}")
    print(f"R2: {100 * metrics.r2_score(y_true, y_predicted): .3f} %")
    print(f"Max Residual Error: {metrics.max_error(y_true, y_predicted): .3f}")
