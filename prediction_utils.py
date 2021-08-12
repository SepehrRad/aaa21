from sklearn import metrics
import math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd


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


def create_prediction_error_line_plt_nn(df, temporal_res, save_fig=True):
    """
    This function plots the average prediction vs. actual demand in different temporal resolution.
    Furthermore, the function saves the result upon request as a png image.
    ----------------------------------------------
    :param
        df (pandas.DataFrame): The given data set
        temporal_resolution (String): the target temporal resolution
        save_fig (boolean): whether to save the image or not
    """
    PLOT_CONST = {
        "D": ["Day", "Daily"],
        "6H": ["Time Zone", "6H"],
        "H": ["Hour", "Hourly"],
    }
    if temporal_res == "H":
        plt.figure(figsize=(30, 10))
    else:
        plt.figure(figsize=(15, 5))
    comparison_plot_data = pd.DataFrame({'Actual': df.groupby(['Date'])[f'Demand ({temporal_res})'].mean(),
                                         'Predictions': df.groupby(['Date'])[
                                             f'Demand ({temporal_res}) Predictions'].mean(),
                                         })
    _ = sns.lineplot(data=comparison_plot_data, markers=True)
    plt.title(f'Average Actual Demand per {PLOT_CONST.get(temporal_res)[0]} vs. Predicted Demand')
    plt.xlabel('Date Time')
    plt.ylabel(f'{PLOT_CONST.get(temporal_res)[1]} Demand')
    plt.show()
    if save_fig:
        _.figure.savefig(f'img/{PLOT_CONST.get(temporal_res)[1]}_avg_pred_actual.png', bbox_inches='tight', dpi=1000)
