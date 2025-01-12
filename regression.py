import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def perform_linear_regression(dataframe, column_name):
    """
    Perform linear regression using the DataFrame index as the independent variable and a specified column as the dependent variable.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be used as the dependent variable.

    Returns:
        dict: A dictionary containing the regression model, coefficients, intercept, predictions, and the equation of the line.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    # Prepare the data
    X = dataframe.index.values.reshape(-1, 1)  # Use DataFrame index as the independent variable
    y = dataframe[column_name].values.reshape(-1, 1)

    # Create and train the model
    model = LinearRegression()
    model.fit(X, y)

    # Make predictions
    predictions = model.predict(X)

    # Form the equation of the line
    slope = model.coef_[0][0]
    intercept = model.intercept_[0]
    equation = f"y = {slope:.2f}x + {intercept:.2f}"

    return {
        'model': model,
        'coefficients': model.coef_,
        'intercept': model.intercept_,
        'predictions': predictions,
        'equation': equation
    }

def plot_regression(dataframe, column_name, regression_results):
    """
    Plot the data and overlay the regression line using the DataFrame index as the x-axis.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be used as the dependent variable.
        regression_results (dict): The output of `perform_linear_regression` function.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    X = dataframe.index.values
    y = dataframe[column_name].values

    # Plot the data points
    plt.scatter(X, y, color='blue', label='Data points')

    # Plot the regression line
    plt.plot(X, regression_results['predictions'], color='red', label='Regression line')

    # Add labels and legend
    plt.xlabel('Index')
    plt.ylabel(column_name)
    plt.title(f"Linear Regression\n{regression_results['equation']}")
    plt.legend()

    plt.show()

def adjust_outliers_linear(dataframe, column_name, regression_results, std_multiplier=2):
    """
    Adjust outliers based on the linear regression line and a multiple of the standard deviation.

    Parameters:
        dataframe (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to be used as the dependent variable.
        regression_results (dict): The output of `perform_linear_regression` function.
        std_multiplier (float): The number of standard deviations above which deviations are considered outliers.

    Returns:
        pd.DataFrame: A DataFrame with adjusted target values.
    """
    if column_name not in dataframe.columns:
        raise ValueError(f"Column '{column_name}' not found in the DataFrame.")

    X = dataframe.index.values.reshape(-1, 1)
    y = dataframe[column_name].values

    # Predicted values based on the regression line
    y_pred = regression_results['model'].predict(X).flatten()

    # Calculate residuals and standard deviation
    residuals = y - y_pred
    std_dev = np.std(residuals)

    # Identify outliers
    threshold = std_multiplier * std_dev
    outliers = np.abs(residuals) > threshold

    # Adjust outliers to the closest point on the regression line
    adjusted_y = np.where(outliers, y_pred, y)

    # Create a new DataFrame with adjusted target values
    adjusted_dataframe = dataframe.copy()
    adjusted_dataframe[column_name] = adjusted_y

    return adjusted_dataframe

if __name__ == "__main__":
    # Example usage
    df = pd.DataFrame({
        'target': [2.2, 4.1, 20.3, 8.4, 10.1]  # Notice the outlier at index 2
    })

    results = perform_linear_regression(df, 'target')
    adjusted_df = adjust_outliers_linear(df, 'target', results, std_multiplier=2)

    print("Original DataFrame:")
    print(df)
    print("\nAdjusted DataFrame:")
    print(adjusted_df)

    # Visualize the original and adjusted data
    plot_regression(df, 'target', results)
    adjusted_results = perform_linear_regression(adjusted_df, 'target')
    plot_regression(adjusted_df, 'target', adjusted_results)
