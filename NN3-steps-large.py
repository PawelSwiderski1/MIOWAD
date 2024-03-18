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


np.random.seed(42)


class MLP:
    def __init__(self, layer_sizes, scaler, verbose=False, beta_momentum=0.90, beta_rmsprop=0.999, epsilon=1e-8):

        self.layer_sizes = layer_sizes
        # Decided to use he method for initialization
        self.weights = [
            np.random.randn(y, x) * np.sqrt(2.0 / x)
            for x, y in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]
        self.scaler = scaler
        self.beta_momentum = beta_momentum
        self.beta_rmsprop = beta_rmsprop
        self.epsilon = epsilon
        # Initialize momentum and RMSprop caches
        self.vdw = [np.zeros_like(w) for w in self.weights]
        self.sdw = [np.zeros_like(w) for w in self.weights]
        self.vdb = [np.zeros_like(b) for b in self.biases]
        self.sdb = [np.zeros_like(b) for b in self.biases]
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
            filepath = f"weights_plots/multimodal-large/{filename}"

            # Save the plot to file
            plt.savefig(filepath)

            plt.clf()

    def print_final_weights_and_biases(self):
        print("Final Weights and Biases:", self.weights, self.biases)

        for i, w in enumerate(self.weights):
            plt.figure(figsize=(40, 20))
            plt.hist(w.flatten(), bins=50)
            plt.title(f"Layer {i + 1} Weight Distribution")
            plt.xlabel("Weight Value")
            plt.ylabel("Frequency")

            filename = f"final_layer_{i + 1}_weights_distribution.png"
            filepath = f"weights_plots/multimodal-large/{filename}"

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

    def real_mse(self, training_data):
        X_scaled = np.array([x for x, y in training_data])
        train_predictions_scaled = np.array(
            [self.feedforward(x.reshape(-1, 1))[0] for x in X_scaled]
        )
        y_true_scaled = np.array([y for x, y in training_data])
        # Inverse transform the predictions and the true y values
        train_predictions = self.scaler.inverse_transform(train_predictions_scaled.reshape(-1, 1))
        y_true = self.scaler.inverse_transform(y_true_scaled)
        # Calculate MSE on the denormalized values
        mse_score_train = mse(train_predictions, y_true)
        return mse_score_train

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

        # Update velocities for weights
        self.vdw = [self.beta_momentum * v + (1 - self.beta_momentum) * nw for v, nw in zip(self.vdw, nabla_w)]
        self.vdb = [self.beta_momentum * v + (1 - self.beta_momentum) * nb for v, nb in zip(self.vdb, nabla_b)]

        # Update squared gradients for weights
        self.sdw = [self.beta_rmsprop * s + (1 - self.beta_rmsprop) * (nw ** 2) for s, nw in zip(self.sdw, nabla_w)]
        self.sdb = [self.beta_rmsprop * s + (1 - self.beta_rmsprop) * (nb ** 2) for s, nb in zip(self.sdb, nabla_b)]

        # Correct the bias for initial iterations for both velocity and squared gradients
        vdw_corrected = [v / (1 - self.beta_momentum ** (i + 1)) for i, v in enumerate(self.vdw)]
        vdb_corrected = [v / (1 - self.beta_momentum ** (i + 1)) for i, v in enumerate(self.vdb)]
        sdw_corrected = [s / (1 - self.beta_rmsprop ** (i + 1)) for i, s in enumerate(self.sdw)]
        sdb_corrected = [s / (1 - self.beta_rmsprop ** (i + 1)) for i, s in enumerate(self.sdb)]

        # Update weights and biases with L2 regularization, RMSprop and Momentum
        self.weights = [(1 - learning_rate * (lambda_ / n)) * w - (learning_rate / len(mini_batch)) * (
                v / (np.sqrt(s) + self.epsilon))
                        for w, v, s in zip(self.weights, vdw_corrected, sdw_corrected)]
        self.biases = [b - (learning_rate / len(mini_batch)) * (v / (np.sqrt(s) + self.epsilon))
                       for b, v, s in zip(self.biases, vdb_corrected, sdb_corrected)]

    def train(
            self,
            training_data,
            epochs,
            learning_rate,
            batch_size,
            test_data=None,
            treshold_mse_train=-(np.inf),
            treshold_mse_test=-(np.inf),
            lambda_=0.0,
            update_method="batch",
            plot_interval=None,
    ):
        n = len(training_data)
        learning_rate_init = learning_rate

        for j in range(epochs):
            if j % (epochs / 100) == 0:
                print("Epoch: ", j)

                mse_train = self.real_mse(training_data)
                print(mse_train)
                if test_data:
                    mse_test = self.real_mse(test_data)
                    if mse_train < treshold_mse_train:
                        print(mse_test)
                        if mse_test < treshold_mse_test:
                            break
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
            #learning_rate = learning_rate_init / (1 + 0.1 * j)
            if j % 10 == 0:
                learning_rate *= 0.8


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


df_train_square = pd.read_csv("mio1/regression/steps-large-training.csv")
X_train_square = df_train_square["x"].values.reshape(-1, 1)
y_train_square = df_train_square["y"].values.reshape(-1, 1)

df_test_square = pd.read_csv("mio1/regression/steps-large-test.csv")
X_test_square = df_test_square["x"].values.reshape(-1, 1)
y_test_square = df_test_square["y"].values.reshape(-1, 1)

# Initialize the scaler for X and y with the desired scaling method
scaler_X = DataScaler(method="min_max")
scaler_y = DataScaler(method="min_max")

# Fit and transform the training data
X_train_scaled = scaler_X.fit_transform(X_train_square)
y_train_scaled = scaler_y.fit_transform(y_train_square)
X_test_scaled = scaler_X.fit_transform(X_test_square)
y_test_scaled = scaler_y.fit_transform(y_test_square)

# Train network on the scaled data
# (set verbose to True if you want more information about the process and to plot weights)

mlp_square_1_5 = MLP([1, 64, 64, 1], scaler=scaler_y, verbose=False)
training_data_scaled = [
    (x.reshape(-1, 1), y) for x, y in zip(X_train_scaled, y_train_scaled)
]
test_data_scaled = [
    (x.reshape(-1, 1), y) for x, y in zip(X_test_scaled, y_test_scaled)
]
mlp_square_1_5.train(training_data_scaled, epochs=500, learning_rate=0.1, batch_size=10, test_data=test_data_scaled,
                     treshold_mse_train=20, treshold_mse_test=3, plot_interval=20)


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
    print("x_test: ", X_test_square[i], "predicted value: ", predictions[i], "actual value: ", y_test_square[i])

mse_score_train = mse(train_predictions, y_train_square)

print(f"Train MSE Score: {mse_score_train}")

mse_score = mse(predictions, y_test_square)

print(f"MSE Score: {mse_score}")

if mlp_square_1_5.verbose:
    mlp_square_1_5.print_final_weights_and_biases()
