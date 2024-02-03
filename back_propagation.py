from utils import NeuralNet, Dataset, preprocess_file, scale, descale
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

def create_neural_net(layers):
    L = len(layers)
    n = layers.copy()

    h = []
    ξ = []
    θ = []
    delta = []
    d_θ = []
    for ℓ in range(L):
        h.append(np.zeros(layers[ℓ]))
        ξ.append(np.zeros(layers[ℓ]))
        θ.append(np.random.rand(layers[ℓ]))  # random values
        delta.append(np.zeros(layers[ℓ]))
        d_θ.append(np.zeros(layers[ℓ]))

    w = []
    d_w = []

    w.append(np.zeros((1, 1)))  # unused, for indexing purposes
    d_w.append(np.zeros((1, 1)))

    for ℓ in range(1, L):
        w.append(np.random.rand(layers[ℓ], layers[ℓ - 1]))  # random values
        d_w.append(np.zeros((layers[ℓ], layers[ℓ - 1])))

    return NeuralNet(L, n, h, ξ, w, θ, delta, d_w, d_θ)

def sigmoid(h):
    return 1 / (1 + np.exp(-h))

def sigmoid_derivative(ξ):
    return ξ * (1 - ξ)

def feed_forward(nn, x_in, y_out):
    nn.ξ[0] = x_in

    for ℓ in range(1, nn.L):
        for i in range(nn.n[ℓ]):
            h = -nn.θ[ℓ][i]
            for j in range(nn.n[ℓ - 1]):
                h += nn.w[ℓ][i, j] * nn.ξ[ℓ - 1][j]
            nn.h[ℓ][i] = h
            nn.ξ[ℓ][i] = sigmoid(h)

    y_out[:] = nn.ξ[-1]

def bp_error(nn, y_out, z):
    for i in range(nn.n[-1]):
        nn.delta[-1][i] = sigmoid_derivative(nn.ξ[-1][i]) * (y_out[i] - z)
    for ℓ in range(nn.L - 1, 0, -1):
        for j in range(nn.n[ℓ - 1]):
            current_error = 0
            for i in range(nn.n[ℓ]):
                current_error += nn.delta[ℓ][i] * nn.w[ℓ][i, j]
            nn.delta[ℓ - 1][j] = sigmoid_derivative(nn.ξ[ℓ - 1][j]) * current_error

def update_threshold_weights(nn, η, α):
    for ℓ in range(1, nn.L):
        for i in range(nn.n[ℓ]):
            for j in range(nn.n[ℓ - 1]):
                nn.d_w[ℓ][i, j] = -η * nn.delta[ℓ][i] * nn.ξ[ℓ - 1][j] + α * nn.d_w[ℓ][i, j]
                nn.w[ℓ][i, j] += nn.d_w[ℓ][i, j]
            nn.d_θ[ℓ][i] = η * nn.delta[ℓ][i] + α * nn.d_θ[ℓ][i]
            nn.θ[ℓ][i] += nn.d_θ[ℓ][i]

def quadratic_error(y_pred, y_true):
    mse = np.abs(y_pred - y_true).sum() / y_true.sum() * 100
    return mse

def back_propagation(nn, data, η, α, epoch):
    print("...Back Propagation()")
    y_out = np.zeros(nn.n[-1])
    y_pred_train = np.zeros(len(data.train))
    y_pred_test = np.zeros(len(data.test))

    mse_train = np.zeros(epoch)
    mse_test = np.zeros(epoch)

    for e in range(epoch):
        for i in range(len(data.train)):
            rnd_num = np.random.choice(len(data.train))
            x_in = data.train[rnd_num, :-1]
            z = data.train[rnd_num, -1]

            feed_forward(nn, x_in, y_out)
            bp_error(nn, y_out, z)
            update_threshold_weights(nn, η, α)

        # Calculate MSE
        for k in range(len(data.train)):
            x_in = data.train[k, :-1]
            feed_forward(nn, x_in, y_out)
            y_pred_train[k] = y_out[0]

        for k in range(len(data.test)):
            x_in = data.test[k, :-1]
            feed_forward(nn, x_in, y_out)
            y_pred_test[k] = y_out[0]

        mse_train[e] = quadratic_error(y_pred_train, data.train[:, -1])
        mse_test[e] = quadratic_error(y_pred_test, data.test[:, -1])

    # Output results
    print(f"Relative absolute error over {epoch} Epochs")
    print(f"Relative absolute error Train: {mse_train[-1]}")
    print(f"Relative absolute error Test: {mse_test[-1]}")

    # Plotting and saving results
    plt.scatter(data.train[:, -1], y_pred_train)
    plt.title("Predicted Vs Original Train")
    plt.xlabel("Original")
    plt.ylabel("Prediction")
    os.makedirs("Plots/BP", exist_ok=True)
    plt.savefig("Plots/BP/figure_Real_Predict_Train.png")

    plt.scatter(data.test[:, -1], y_pred_test)
    plt.title("Predicted Vs Original Test")
    plt.xlabel("Original")
    plt.ylabel("Prediction")
    plt.savefig("Plots/BP/figure_Real_Predict_Test.png")

    plt.plot(mse_train)
    plt.title("Training % Error over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("%Error")
    plt.savefig("Plots/BP/figure_Error_Train.png")

    plt.plot(mse_test)
    plt.title("Test % Error over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("%Error")
    plt.savefig("Plots/BP/figure_Error_Test.png")

    return np.sum(mse_test) / epoch
