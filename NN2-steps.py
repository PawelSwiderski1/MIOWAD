import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


def cost_derivative(output_activations, y):
    return output_activations - y


np.random.seed(41)


class MLP:
    def __init__(self, layer_sizes, verbose=False):
        self.layer_sizes = layer_sizes
        # Decided to use he method for initialization
        self.weights = [
            np.random.randn(y, x) * np.sqrt(2.0 / x)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.verbose = verbose
        if self.verbose:
            print("Initial weights: ", self.weights)

    def plot_weights(self, epoch):
        for i, w in enumerate(self.weights):
            plt.figure(figsize=(40, 20))
            plt.hist(w.flatten(), bins=50)
            plt.title(f"Layer {i + 1} Weight Distribution")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")

            filename = f"epoch_{epoch}_layer_{i + 1}_weights_distribution.png"
            filepath = f"weights_plots/steps-small/{filename}"

            # Save the plot to file
            plt.savefig(filepath)

            plt.clf()

    def print_final_weights_and_biases(self):
        if self.verbose:
            print("Final Weights and Biases:", self.weights, self.biases)

            for i, w in enumerate(self.weights):

                plt.figure(figsize=(40, 20))
                plt.hist(w.flatten(), bins=50)
                plt.title(f"Layer {i + 1} Weight Distribution")
                plt.xlabel("Weight Value")
                plt.ylabel("Frequency")

                filename = f"final_layer_{i + 1}_weights_distribution.png"
                filepath = f"weights_plots/steps-small/{filename}"

                # Save the plot to file
                plt.savefig(filepath)

                plt.clf()

    def feedforward(self, a):
        activations = [a]  # Stores all activations
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = relu(np.dot(w, a) + b)
            activations.append(a)
        # Linear activation for the last layer
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        activations.append(a)
        return (
            activations[-1],
            activations,
        )  # Return final activation and all activations

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        final_output, activations = self.feedforward(x)
        zs = [
            np.dot(w, act) + b
            for w, b, act in zip(self.weights, self.biases, activations[:-1])
        ]  # Z values

        # Output layer error
        delta = cost_derivative(final_output, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate the error
        for l in range(2, len(self.layer_sizes)):
            sp = relu_derivative(zs[-l])
            delta = np.dot(self.weights[-l + 1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l - 1].T)

        return nabla_w, nabla_b

    def update_mini_batch(self, mini_batch, learning_rate, lambda_, n):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        for x, y in mini_batch:
            delta_nabla_w, delta_nabla_b = self.backprop(x, y)
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]

        # Update weights with L2 regularization
        self.weights = [
            (1 - learning_rate * (lambda_ / n)) * w
            - (learning_rate / len(mini_batch)) * nw
            for w, nw in zip(self.weights, nabla_w)
        ]
        self.biases = [
            b - (learning_rate / len(mini_batch)) * nb
            for b, nb in zip(self.biases, nabla_b)
        ]

    def train(
        self,
        training_data,
        epochs,
        learning_rate,
        batch_size,
        lambda_=0.0,
        update_method="batch",
        plot_interval=None,
    ):
        n = len(training_data)
        learning_rate_init = learning_rate

        for j in range(epochs):

            if j % (epochs / 10) == 0:
                print("Epoch: ", j)
            # Plot weights at the specified interval
            if self.verbose and plot_interval and j % plot_interval == 0:
                self.plot_weights(epoch=j)

            np.random.shuffle(training_data)
            if update_method == "batch":
                mini_batches = [
                    training_data[k: k + batch_size] for k in range(0, n, batch_size)
                ]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, learning_rate, lambda_, n)
            elif update_method == "epoch":
                self.update_mini_batch(training_data, learning_rate, lambda_, n)
            # Learning rate schedule
            learning_rate = learning_rate_init / (1 + 0.01 * j)


class DataScaler:
    def __init__(self, method="standardization"):
        self.method = method
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit_transform(self, data):
        if self.method == "min_max":
            return self.fit_transform_min_max(data)
        elif self.method == "standardization":
            return self.fit_transform_standardization(data)
        else:
            raise ValueError("Unsupported scaling method")

    def transform(self, data):
        if self.method == "min_max":
            return self.transform_min_max(data)
        elif self.method == "standardization":
            return self.transform_standardization(data)
        else:
            raise ValueError("Unsupported scaling method")

    def inverse_transform(self, data):
        if self.method == "min_max":
            return self.inverse_transform_min_max(data)
        elif self.method == "standardization":
            return self.inverse_transform_standardization(data)
        else:
            raise ValueError("Unsupported scaling method")

    def fit_transform_min_max(self, data):
        self.min = np.min(data, axis=0)
        self.max = np.max(data, axis=0)
        return (data - self.min) / (self.max - self.min)

    def transform_min_max(self, data):
        return (data - self.min) / (self.max - self.min)

    def inverse_transform_min_max(self, data):
        return data * (self.max - self.min) + self.min

    def fit_transform_standardization(self, data):
        self.mean = np.mean(data, axis=0)
        self.std = np.std(data, axis=0)
        return (data - self.mean) / self.std

    def transform_standardization(self, data):
        return (data - self.mean) / self.std

    def inverse_transform_standardization(self, data):
        return data * self.std + self.mean


df_train_square = pd.read_csv("mio1/regression/steps-small-training.csv")
X_train_square = df_train_square["x"].values.reshape(-1, 1)
y_train_square = df_train_square["y"].values.reshape(-1, 1)

# Initialize the scaler for X and y with the desired scaling method
scaler_X = DataScaler(method="standardization")
scaler_y = DataScaler(method="standardization")

# Fit and transform the training data
X_train_scaled = scaler_X.fit_transform(X_train_square)
y_train_scaled = scaler_y.fit_transform(y_train_square)

# Train network on the scaled data
# (set verbose to True if you want more information about the process and to plot weights)

mlp_square_1_5 = MLP([1, 10, 5, 5, 1], verbose=False)
training_data_scaled = [
    (x.reshape(-1, 1), y) for x, y in zip(X_train_scaled, y_train_scaled)
]
mlp_square_1_5.train(training_data_scaled, epochs=1500, learning_rate=0.1, batch_size=1, plot_interval=150)

df_test_square = pd.read_csv("mio1/regression/steps-small-test-mod.csv")
X_test_square = df_test_square["x"].values.reshape(-1, 1)
y_test_square = df_test_square["y"].values.reshape(-1, 1)

# Scale the test data using the transform method
X_test_scaled = scaler_X.transform(X_test_square)

# Generate predictions on the scaled test data and inverse transform
predictions_scaled = np.array(
    [mlp_square_1_5.feedforward(x.reshape(-1, 1))[0] for x in X_test_scaled]
)
predictions = scaler_y.inverse_transform(predictions_scaled.reshape(-1, 1))

train_predictions_scaled = np.array(
    [mlp_square_1_5.feedforward(x.reshape(-1, 1))[0] for x in X_train_scaled]
)
train_predictions = scaler_y.inverse_transform(train_predictions_scaled.reshape(-1, 1))

# Calculate and print the MSE score
for i in range(len(predictions)):
    print("x_test: ", X_test_square[i], "predicted value: ", predictions[i],"actual value: ", y_test_square[i])

mse_score_train = mse(train_predictions, y_train_square)

print(f"Train MSE Score: {mse_score_train}")

mse_score = mse(predictions, y_test_square)

print(f"MSE Score: {mse_score}")

if mlp_square_1_5.verbose:
    mlp_square_1_5.print_final_weights_and_biases()
