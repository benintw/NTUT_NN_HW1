# Re-run the code without the 'icecream' library as it's not available in this environment

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
LEARNING_RATE = 0.3
EPOCHS = 50
BIAS = -1

# 3D input data (x1, x2, x3)
X = np.array(
    [[-0.5, -0.5, 0.2], [-0.5, 0.5, -0.3], [0.3, -0.5, 0.8], [-0.1, 1.0, -0.5]]
)
y = np.array([1, 1, -1, -1])

# Initialize weights for the 3D space (small random initialization)
weights = np.random.randn(X.shape[1]) * 0.01

# Lists to store errors and weights at each iteration and epoch
errors_per_epoch = []
errors_per_iteration = []
weights_per_epoch = []

# Perceptron Training Process
iteration = 0
for epoch in range(EPOCHS):
    errors = 0  # Track errors in this epoch
    for inputs, target in zip(X, y):
        iteration += 1

        # Net input (linear combination of inputs and weights)
        net_input = np.dot(inputs, weights) - BIAS

        # Prediction using sign activation function
        prediction = np.sign(net_input)

        # Update weights if there is a misclassification
        if prediction != target:
            weights = weights + LEARNING_RATE * (target - prediction) * inputs
            errors += 1

        # Track error per iteration
        errors_per_iteration.append(errors)

    # Track weights per epoch and errors per epoch
    weights_per_epoch.append(weights.copy())
    errors_per_epoch.append(errors)

    # Stop if no errors
    if errors == 0:
        print(f"Perceptron converged at epoch {epoch}")
        break

# Plot 3D Decision Planes over Epochs
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")

# Meshgrid for plotting the plane
x1_range = np.linspace(-0.6, 0.6, 10)
x2_range = np.linspace(-0.6, 0.6, 10)
x1_mesh, x2_mesh = np.meshgrid(x1_range, x2_range)

# Plot decision planes for each epoch
for i, epoch_weights in enumerate(weights_per_epoch):

    if i < len(weights_per_epoch) - 1:
        if not np.isclose(
            epoch_weights[2], 0
        ):  # Avoid division by zero in x3 calculation
            x3_mesh = (
                BIAS - epoch_weights[0] * x1_mesh - epoch_weights[1] * x2_mesh
            ) / epoch_weights[2]
            ax.plot_surface(x1_mesh, x2_mesh, x3_mesh, alpha=0.1)

if not np.isclose(weights[2], 0):
    x3_mesh = (BIAS - weights[0] * x1_mesh - weights[1] * x2_mesh) / weights[2]
    ax.plot_surface(x1_mesh, x2_mesh, x3_mesh, alpha=0.8, label="Final Decision Plane")


# Plot data points
ax.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    X[y == 1][:, 2],
    c="yellow",
    marker="^",
    s=200,
    # label="Class 1",
)
ax.scatter(
    X[y == -1][:, 0],
    X[y == -1][:, 1],
    X[y == -1][:, 2],
    c="red",
    marker="o",
    s=200,
    # label="Class -1",
)

ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.set_zlabel("x3")
ax.set_title("Perceptron Decision Planes in 3D Over Epochs")
ax.legend()
plt.show()

# Plot Error vs. Iteration
plt.figure(figsize=(8, 6))
plt.plot(errors_per_iteration, marker="o")
plt.title("Error vs. Iteration (Perceptron)")
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.grid(True)
plt.show()

# Plot Error vs. Epoch
plt.figure(figsize=(8, 6))
plt.plot(errors_per_epoch, marker="o", color="red")
plt.title("Error vs. Epoch (Perceptron)")
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.grid(True)
plt.show()
