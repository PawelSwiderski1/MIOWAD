import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


np.random.seed(42)  # That was for the mse 6



class MLP:
    def __init__(self, layer_sizes):
        self.layer_sizes = layer_sizes
        #self.weights = [np.random.randn(y, x) * np.sqrt(2. / x)
        #                  for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.weights = [np.random.rand(y, x) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.biases = [np.random.randn(y, 1) for y in layer_sizes[1:]]

    def plot_weights(self, epoch):
        for i, w in enumerate(self.weights):
            plt.figure(figsize=(40, 20))
            plt.hist(w.flatten(), bins=50)
            plt.title(f'Layer {i + 1} Weight Distribution')
            plt.xlabel('Weight Value')
            plt.ylabel('Frequency')

            # Construct the filename for the plot
            filename = f'epoch_{epoch}_layer_{i + 1}_weights_distribution_wrong.png'
            filepath = f'weights_plots/{filename}'

            # Save the plot to file
            plt.savefig(filepath)

            # Optionally, clear the current figure to free memory for the next plot
            plt.clf()

    def print_final_weights_and_biases(self):
        print("Final Weights and Biases:")
        # for i, (w, b) in enumerate(zip(self.weights, self.biases)):
        #     print(f"Layer {i + 1} Weights:\n{w}")
        #     print(f"Layer {i + 1} Biases:\n{b}")
        print(self.weights)
        print(self.biases)

    def feedforward(self, a):
        activations = [a]  # Stores all activations
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
            activations.append(a)
        # Linear activation for the last layer
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        activations.append(a)
        return activations[-1], activations  # Return final activation and all activations

    def backprop(self, x, y):
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        final_output, activations = self.feedforward(x)
        zs = [np.dot(w, act) + b for w, b, act in zip(self.weights, self.biases, activations[:-1])]  # Z values

        # Output layer error
        delta = self.cost_derivative(final_output, y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        # Backpropagate the error
        for l in range(2, len(self.layer_sizes)):
            sp = sigmoid_derivative(zs[-l])
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
        self.weights = [(1 - learning_rate * (lambda_ / n)) * w - (learning_rate / len(mini_batch)) * nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (learning_rate / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]
        # print("Weights:", self.weights)
        # print("Biases:", self.biases)

    def train(self, training_data, epochs, learning_rate, batch_size, lambda_=0.0, update_method='batch',
              plot_interval=None):
        n = len(training_data)
        learning_rate_init = learning_rate
        for j in range(epochs):
            # Plot weights at the specified interval
            if plot_interval and j % plot_interval == 0:
                print(f"Epoch {j}:")
                #self.plot_weights(epoch=j)

            np.random.shuffle(training_data)
            if update_method == 'batch':
                mini_batches = [training_data[k:k + batch_size] for k in range(0, n, batch_size)]
                for mini_batch in mini_batches:
                    self.update_mini_batch(mini_batch, learning_rate, lambda_, n)
            elif update_method == 'epoch':
                self.update_mini_batch(training_data, learning_rate, lambda_, n)
            # Learning rate schedule
            learning_rate = learning_rate_init / (1 + 0.01 * j)

    def cost_derivative(self, output_activations, y):
        return (output_activations - y)


class DataScaler:
    def __init__(self):
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

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


df_train_square = pd.read_csv('mio1/regression/square-simple-training.csv')
X_train_square = df_train_square['x'].values.reshape(-1, 1)
y_train_square = df_train_square['y'].values.reshape(-1, 1)

# Initialize the scaler
scaler_X = DataScaler()
scaler_y = DataScaler()

# Choose between min-max normalization and standardization
#X_train_scaled = scaler_X.fit_transform_min_max(X_train_square)  # For min-max normalization
X_train_scaled = scaler_X.fit_transform_standardization(X_train_square)  # For standardization

#y_train_scaled = scaler_y.fit_transform_min_max(y_train_square)  # For min-max normalization
y_train_scaled = scaler_y.fit_transform_standardization(y_train_square)  # For standardization

# Train your network on the scaled data
mlp_square_1_5 = MLP([1, 10, 1])
training_data_scaled = [(x.reshape(-1, 1), y) for x, y in zip(X_train_scaled, y_train_scaled)]
mlp_square_1_5.train(training_data_scaled, epochs=3000, learning_rate=1, batch_size=10, plot_interval=100)

df_test_square = pd.read_csv('mio1/regression/square-simple-test.csv')
X_test_square = df_test_square['x'].values.reshape(-1, 1)
y_test_square = df_test_square['y'].values.reshape(-1, 1)

# Correctly scale test data (DO NOT refit scaler)
#X_test_scaled = scaler_X.transform_min_max(X_test_square)  # For min-max normalization
X_test_scaled = scaler_X.transform_standardization(X_test_square)  # For standardization

# Generate predictions on the scaled test data
predictions_scaled = np.array([mlp_square_1_5.feedforward(x.reshape(-1, 1))[0] for x in X_test_scaled])

# Correctly denormalize predictions
#predictions = scaler_y.inverse_transform_min_max(predictions_scaled.reshape(-1, 1))  # For min-max normalization
predictions = scaler_y.inverse_transform_standardization(predictions_scaled.reshape(-1, 1))  # For standardization

# Calculate MSE score
for i in range(len(predictions)):
    print(predictions[i], y_test_square[i])
mse_score = mse(predictions, y_test_square)

print(f"MSE Score: {mse_score}")

mlp_square_1_5.print_final_weights_and_biases()
