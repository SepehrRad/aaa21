from sklearn import metrics
import math


def get_prediction_scores(y_true, y_predicted):
    """
    This function prints the prediction scores.
    ----------------------------------------------
    :param
        y_true (List): Test values
        y_true (List): Predicted values
    """
    print("-------MODEL SCORES-------")
    print(f"MAE: {metrics.mean_absolute_error(y_true, y_predicted): .3f}")
    print(f"MSE: {metrics.mean_squared_error(y_true, y_predicted): .3f}")
    print(f"RMSE: {math.sqrt(metrics.mean_squared_error(y_true, y_predicted)): .3f}")
    print(f"R2: {100 * metrics.r2_score(y_true, y_predicted): .3f} %")
