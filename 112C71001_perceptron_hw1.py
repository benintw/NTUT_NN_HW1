import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

# Parameters
LEARNING_RATE = 0.3
EPOCHS = 50
BIAS = -1


X = np.array([[-0.5, -0.5], [-0.5, 0.5], [0.3, -0.5], [-0.1, 1.0]])
y = np.array([1, 1, -1, -1])

# Initialize weights Small random initialization
weights = np.random.randn(X.shape[1]) * 0.01

# lists to store errors and weights at each iteration and epoch
errors_per_epoch = []
errors_per_iteration = []
weights_per_epoch = []

# Perceptron Training Process
iteration = 0
for epoch in range(EPOCHS):
    errors = 0  # track errors in this epoch
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

        # track error per iteration
        errors_per_iteration.append(errors)

    # track weights per epoch and errors per epoch
    weights_per_epoch.append(weights.copy())
    errors_per_epoch.append(errors)

    # Stop if no errors
    if errors == 0:
        print(f"Perceptron converged at epoch {epoch}")
        break
# Plot 1: Decision Boundary Lines for each epoch
plt.figure(figsize=(8, 6))
x_min, x_max = -0.6, 0.6
x1_values = np.linspace(x_min, x_max, 100)

for i, epoch_weights in enumerate(weights_per_epoch):
    if i < len(weights_per_epoch) - 1:
        # Decision boundary: x2 = (bias - w1*x1) / w2
        if not np.isclose(epoch_weights[1], 0):
            decision_boundary = (BIAS - epoch_weights[0] * x1_values) / epoch_weights[1]
            plt.plot(
                x1_values,
                decision_boundary,
                linestyle="--",
            )

# Plot final decision boundary
if not np.isclose(weights[1], 0):
    final_decision_boundary = (BIAS - weights[0] * x1_values) / weights[1]
    plt.plot(x1_values, final_decision_boundary, "o-", label="Final Decision Boundary")

# Plot data points
plt.scatter(
    X[y == 1][:, 0],
    X[y == 1][:, 1],
    c="yellow",
    edgecolors="k",
    marker="o",
    s=100,
    label="Class 1",
)
plt.scatter(
    X[y == -1][:, 0],
    X[y == -1][:, 1],
    c="blue",
    edgecolors="k",
    marker="o",
    s=100,
    label="Class -1",
)
plt.xlim(x_min, x_max)
plt.ylim(X[:, 1].min() - 0.5, X[:, 1].max() + 0.5)
plt.title("Perceptron Decision Boundaries per Epoch")
plt.xlabel("x1")
plt.ylabel("x2")
plt.legend()
plt.grid(True)
plt.show()

# Plot 2: Error vs. Iteration
plt.figure(figsize=(8, 6))
plt.plot(errors_per_iteration, marker="o")
plt.title("Error vs. Iteration (Perceptron)")
plt.xlabel("Iteration")
plt.ylabel("Errors")
plt.grid(True)
plt.show()

# Plot 3: Error vs. Epoch
plt.figure(figsize=(8, 6))
plt.plot(errors_per_epoch, marker="o", color="red")
plt.title("Error vs. Epoch (Perceptron)")
plt.xlabel("Epoch")
plt.ylabel("Errors")
plt.grid(True)
plt.show()


ic(weights)
