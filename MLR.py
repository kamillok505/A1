from utils import Dataset, get_ranges, scale, descale
from back_propagation import back_propagation, create_neural_net
import pandas as pd
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
def mape(y_actual, y_pred):
    return np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100

def mlr(data):
    print("...MLR()")
    response = data.names[-1]
    predictors = data.names[:-1]
    X = data.train_df[predictors]
    y = data.train_df[response]
    X = sm.add_constant(X)  # adding a constant
    linear_regressor = sm.OLS(y, X).fit()
    print(linear_regressor.summary())

    # Prediction
    X_test = sm.add_constant(data.test_df[predictors])
    prediction_test = linear_regressor.predict(X_test)
    X_train = sm.add_constant(data.train_df[predictors])
    prediction_train = linear_regressor.predict(X_train)

    # Test Performance DataFrame (compute squared error)
    performance_test = pd.DataFrame({'y_actual': data.test_df[response], 'y_predicted': prediction_test})
    performance_test['error'] = performance_test['y_actual'] - performance_test['y_predicted']
    performance_test['error_sq'] = performance_test['error'] ** 2

    # Test Error
    print("Mean Absolute test percentage error: ", mape(performance_test['y_actual'], performance_test['y_predicted']), "%\n")

    # Save results
    performance_test_csv = data.test_df.copy()
    performance_test_csv['y_predicted'] = prediction_test
    descaled_performance_test_csv = descale(performance_test_csv, data.rangesTest)
    descaled_performance_test_csv.to_csv("./Results/MLR/_results_test.csv", index=False)

    # Plots
    plt.scatter(descaled_performance_test_csv[response], descaled_performance_test_csv['y_predicted'])
    plt.title("Predicted Vs Original Test")
    plt.xlabel("Original")
    plt.ylabel("Prediction")
    plt.savefig("Plots/MLR/figure_Real_Predict_Test.png")

    return mape(performance_test['y_actual'], performance_test['y_predicted'])

