import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# Parameters
LEARNING_RATE = 0.2
EPOCHS = 200
EPSILON = 1e-5  # Convergence criterion for MSE


X = np.array([[-1, -0.5, -0.5], [-1, -0.5, 0.5], [-1, 0.3, -0.5], [-1, -0.1, 1.0]])
y = np.array([1, 1, -1, -1])

# Initialize weights (including the bias weight)
weights = np.random.rand(X.shape[1]) * 0.01

mse_per_epoch = []  # To store mean squared errors over epochs
loss_per_iteration = []
weights_per_epoch = []

# ADALINE Training process
for epoch in range(EPOCHS):
    total_error = 0  # Sum of squared errors for this epoch

    for inputs, target in zip(X, y):
        # Net input (linear combination of inputs and weights)
        net_input = np.dot(inputs, weights)

        # Error (difference between desired output and actual output)
        error = target - net_input

        # Update weights using the delta rule
        weights = weights + LEARNING_RATE * error * inputs

        squared_error = error**2
        loss_per_iteration.append(squared_error)

        # Accumulate squared error
        total_error += error**2

    # Calculate mean squared error (MSE) for this epoch
    mse = total_error / X.shape[0]
    mse_per_epoch.append(mse)

    weights_per_epoch.append(weights.copy())

    # Stopping criterion based on MSE change
    if epoch > 0 and abs(mse_per_epoch[-1] - mse_per_epoch[-2]) <= EPSILON:
        converged_epoch = epoch
        print(f"Converged at epoch {epoch}")
        break


# ic(weights_per_epoch)
ic(weights)


# Plot 1: Mean Squared Error (MSE) over epochs
plt.figure(figsize=(8, 6))
plt.plot(mse_per_epoch, marker="o", color="red")
plt.title("Mean Squared Error Over Epochs (ADALINE)")
plt.xlabel("Epoch")
plt.ylabel("Mean Squared Error (MSE)")
plt.grid(True)

# Add text for convergence if applicable
if "converged_epoch" in locals():
    plt.text(
        converged_epoch,
        mse_per_epoch[converged_epoch] + 0.2,
        f"Converged at epoch {converged_epoch}",
        fontsize=12,
        color="green",
        verticalalignment="bottom",
        horizontalalignment="right",
    )

plt.show()

# Plot 2: Loss (Squared Error) per iteration
plt.figure(figsize=(8, 6))
plt.plot(loss_per_iteration, marker="o")
plt.title("Loss (Squared Error) Per Iteration (ADALINE)")
plt.xlabel("Iteration")
plt.ylabel("Squared Error")
plt.grid(True)
plt.show()

# Plot 3: Decision Boundary and Input Data
plt.figure(figsize=(8, 6))
x_min, x_max = -0.6, 0.6
x1_values = np.linspace(x_min, x_max, 100)

# Boundary line: x2 = (BIAS - w1*x1) / w2
for i, epoch_weights in enumerate(weights_per_epoch):
    if i < len(weights_per_epoch) - 1:
        # Decision boundary: x2 = (w0 - w1*x1) / w2
        if not np.isclose(epoch_weights[2], 0):
            decision_boundary = (epoch_weights[0] - epoch_weights[1] * x1_values) / epoch_weights[2]
            plt.plot(
                x1_values,
                decision_boundary,
                linestyle="--",
            )

# Plot final decision boundary
if not np.isclose(weights[2], 0):  # Make sure the denominator is not zero
    decision_boundary = (weights[0] - weights[1] * x1_values) / weights[2]
    plt.plot(x1_values, decision_boundary, "o-", label="Final Decision Boundary")

# Plot data points
plt.scatter(
    X[y == 1][:, 1],
    X[y == 1][:, 2],
    c="yellow",
    edgecolors="k",
    marker="o",
    s=100,
    label="Class 1",
)
plt.scatter(
    X[y == -1][:, 1],
    X[y == -1][:, 2],
    c="blue",
    edgecolors="k",
    marker="o",
    s=100,
    label="Class -1",
)

# Set limits and labels
# plt.xlim(x_min, x_max)
plt.xlim(-5, 5)
# plt.ylim(X[:, 2].min() - 0.5, X[:, 2].max() + 0.5)
plt.ylim(-5, 5)
plt.title("Decision Boundary with Input Data")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()
