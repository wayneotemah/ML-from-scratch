import numpy as np


def MSE(y_true, y_pred):
    """
    Calculate the Mean Squared Error (MSE) for regression.

    Parameters:
        y_true (numpy array): Ground truth labels.
        y_pred (numpy array): Predicted labels.

    Returns:
        float: The Mean Squared Error (MSE).
    """
    # Calculate the squared differences between predicted and true values.
    squared_diff = np.square(y_true - y_pred)

    # Calculate the mean of the squared differences.
    mse = np.mean(squared_diff)

    return mse


def MAE(y_true, y_pred):
    """
    Calculate the Mean Absolute Error (MAE) for regression.

    Parameters:
        y_true (numpy array): Ground truth labels.
        y_pred (numpy array): Predicted labels.

    Returns:
        float: The Mean Absolute Error (MAE).
    """
    # Calculate the absolute differences between predicted and true values.
    absolute_diff = np.abs(y_true - y_pred)

    # Calculate the mean of the absolute differences.
    mae = np.mean(absolute_diff)

    return mae
