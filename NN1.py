import numpy as np
import pandas as pd


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def mse(predictions, targets):
    return np.mean((predictions - targets) ** 2)


class MLP:
    def __init__(self, layer_sizes, weights, biases):
        self.layer_sizes = layer_sizes
        self.weights = weights
        print(self.weights)
        self.biases = biases

    def feedforward(self, a):
        activations = [a]  # Stores all activations
        for b, w in zip(self.biases[:-1], self.weights[:-1]):
            a = sigmoid(np.dot(w, a) + b)
            activations.append(a)
        # Linear activation for the last layer
        a = np.dot(self.weights[-1], a) + self.biases[-1]
        activations.append(a)
        return activations[-1], activations  # Return final activation and all activations


# SQUARE SIMPLE

weights = [np.array([[-2.54942087],
       [-7.34844221],
       [ 5.66564003],
       [ 9.53355062],
       [-7.01508122]]), np.array([[ -0.17622762,   6.46607767,  -1.73403919,  -9.8232896 ,
          5.05751039],
       [ 15.41458997,  -3.11079245,   2.84169739,   3.3700297 ,
         -2.49115115],
       [-16.41961746,   2.87882694,  -5.95882206,  -6.04112884,
          4.5638308 ],
       [  1.23799655,   5.96920534,  -3.85447961,  -8.07376189,
          5.18789661],
       [ 18.66770321,  -5.52870688,   6.06619413,   4.82976879,
         -5.13622883]]), np.array([[-57.2694563 ,  83.65861736, -55.64452185, -57.08448128,
         91.94495515]])]

biases = [np.array([[ -3.12784542],
       [ 12.21773855],
       [ -2.29184127],
       [-18.85043553],
       [  7.82582976]]), np.array([[ 2.29814006],
       [ 2.90077046],
       [-1.00267574],
       [ 0.83503552],
       [-0.45035457]]), np.array([[29.47562796]])]


mlp_square_2_5 = MLP([1, 5, 5, 1], weights=weights, biases=biases)

df_test_square = pd.read_csv('mio1/regression/square-simple-test.csv')
X_test_square = df_test_square['x'].values.reshape(-1, 1)
y_test_square = df_test_square['y'].values.reshape(-1, 1)

# Generate predictions
predictions = np.array([mlp_square_2_5.feedforward(x.reshape(-1, 1))[0] for x in X_test_square])

# Flatten predictions to ensure it has the same shape as y_test
predictions = predictions.reshape(-1, 1)

# Calculate MSE score
for i in range(len(predictions)):
    print(predictions[i], y_test_square[i])
mse_score = mse(predictions, y_test_square)

print(f"MSE Score: {mse_score}")


# STEPS LARGE

weigths = [np.array([[ 68.92862219],
       [-84.99513297],
       [119.23495792],
       [ 76.89301184],
       [-87.21358006]]), np.array([[ 1.91289906, -0.42319819, 18.07070798,  3.00123577, -1.92057603],
       [ 1.85667133, -2.3699089 , 22.79272698,  3.23949396, -3.90147739],
       [10.32070981,  0.51598607, -8.60220243, 11.48614926, -1.64366081],
       [-1.14679355, 17.30055415, -1.69969307, -3.66070814, 17.40682448],
       [13.95014667, -2.87680419, -4.92717336, 15.68627572, -2.98792556]]), np.array([[ 44.09132003,  45.01472659,  40.25650334, -74.36851247,
         39.73410003]])]

biases = [np.array([[-103.05945909],
       [ -42.70583506],
       [ -60.43845255],
       [-114.88956424],
       [ -43.75453706]]), np.array([[ -5.82400193],
       [ -7.63091442],
       [ -1.24066723],
       [-14.10986627],
       [-10.58238982]]), np.array([[-9.09792412]])]

mlp_steps_2_5 = MLP([1, 5, 5, 1], weights=weigths, biases=biases)

df_test_steps = pd.read_csv('mio1/regression/steps-large-test.csv')
X_test_steps = df_test_steps['x'].values.reshape(-1, 1)
y_test_steps = df_test_steps['y'].values.reshape(-1, 1)

# Generate predictions
predictions = np.array([mlp_steps_2_5.feedforward(x.reshape(-1, 1))[0] for x in X_test_steps])

# Flatten predictions to ensure it has the same shape as y_test
predictions = predictions.reshape(-1, 1)

# Calculate MSE score
for i in range(len(predictions)):
    print(predictions[i], y_test_steps[i])
mse_score = mse(predictions, y_test_steps)

print(f"MSE Score: {mse_score}")

