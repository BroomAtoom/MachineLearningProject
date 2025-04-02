import numpy as np

N = 1000

# Generate random input data (replace these with your actual data)
np.random.seed(12)
x1, x2, x3, x4, x5 = [np.random.rand(N) for _ in range(5)]
y = np.random.randint(0, 2, size=(N, 1))  # Ensure y is a column vector

# Stack inputs into a single matrix
X = np.vstack([x1, x2, x3, x4, x5]).T

# Split dataset (60% training, 20% testing, 20% validation)
n_train = int(0.6 * len(X))
n_test = int(0.2 * len(X))

X_train, y_train = X[:n_train], y[:n_train]
X_test, y_test = X[n_train:n_train + n_test], y[n_train:n_train + n_test]
X_val, y_val = X[n_train + n_test:], y[n_train + n_test:]

# Define the neural network architecture
num_layers = 100  # Change this to set the number of layers
hidden_units = 50  # Number of neurons per hidden layer
input_dim = X.shape[1]

# Initialize weights and biases for all layers
layers = []
prev_dim = input_dim  # First layer input size
for i in range(num_layers):
    layer = {
        "W": np.random.randn(prev_dim, hidden_units) * 0.01,  # Weights
        "b": np.zeros((1, hidden_units))  # Biases
    }
    layers.append(layer)
    prev_dim = hidden_units

# Output layer
output_layer = {
    "W": np.random.randn(prev_dim, 1) * 0.01,
    "b": np.zeros((1, 1))
}
layers.append(output_layer)

# Activation function: tanh
def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2

# Accuracy function
def accuracy(y_true, y_pred):
    y_pred_binary = (y_pred >= 0).astype(int)  # Convert tanh output to 0 or 1
    return np.mean(y_true == y_pred_binary)  # Compare with actual y values

# Training parameters
learning_rate = 0.01
epochs = 2000

# Training loop
for epoch in range(epochs):
    # Forward pass
    A = X_train
    activations = [A]  # Store activations for backpropagation

    for layer in layers:
        Z = np.dot(A, layer["W"]) + layer["b"]
        A = tanh(Z)
        activations.append(A)

    # Compute loss (Mean Squared Error)
    loss = np.mean((A - y_train) ** 2)

    # Compute accuracy
    train_acc = accuracy(y_train, A)

    # Backpropagation
    dA = 2 * (A - y_train) / len(y_train)  # Gradient of MSE

    for i in reversed(range(len(layers))):
        Z = np.dot(activations[i], layers[i]["W"]) + layers[i]["b"]
        dZ = dA * tanh_derivative(Z)

        dW = np.dot(activations[i].T, dZ)
        db = np.sum(dZ, axis=0, keepdims=True)

        # Update weights
        layers[i]["W"] -= learning_rate * dW
        layers[i]["b"] -= learning_rate * db

        dA = np.dot(dZ, layers[i]["W"].T)  # Propagate the gradient backward

    # Print loss and accuracy every 100 epochs
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.5f}, Train Accuracy = {train_acc:.5f}")

# --- Testing the model ---
A_test = X_test
for layer in layers:
    Z_test = np.dot(A_test, layer["W"]) + layer["b"]
    A_test = tanh(Z_test)

test_loss = np.mean((A_test - y_test) ** 2)
test_acc = accuracy(y_test, A_test)
print(f"Test Loss: {test_loss:.5f}, Test Accuracy: {test_acc:.5f}")

# --- Validation ---
A_val = X_val
for layer in layers:
    Z_val = np.dot(A_val, layer["W"]) + layer["b"]
    A_val = tanh(Z_val)

val_loss = np.mean((A_val - y_val) ** 2)
val_acc = accuracy(y_val, A_val)
print(f"Validation Loss: {val_loss:.5f}, Validation Accuracy: {val_acc:.5f}")
