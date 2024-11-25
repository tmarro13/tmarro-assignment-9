import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from matplotlib.patches import Circle

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(42)  # Different seed for variety
        self.lr = lr  # Learning rate
        self.activation_fn = activation  # Activation function

        # Initialize weights and biases
        self.weights_input_hidden = np.random.randn(input_dim, hidden_dim) * 0.1
        self.bias_hidden = np.zeros((1, hidden_dim))
        self.weights_hidden_output = np.random.randn(hidden_dim, output_dim) * 0.1
        self.bias_output = np.zeros((1, output_dim))

        # Store intermediate activations
        self.hidden_layer_output = None

    def activation(self, x):
        """Apply the activation function."""
        if self.activation_fn == 'tanh':
            return np.tanh(x)
        elif self.activation_fn == 'relu':
            return np.maximum(0, x)
        elif self.activation_fn == 'sigmoid':
            return 1 / (1 + np.exp(-x))

    def activation_derivative(self, x):
        """Compute the derivative of the activation function."""
        if self.activation_fn == 'tanh':
            return 1 - np.tanh(x) ** 2
        elif self.activation_fn == 'relu':
            return np.where(x > 0, 1, 0)
        elif self.activation_fn == 'sigmoid':
            sigmoid = 1 / (1 + np.exp(-x))
            return sigmoid * (1 - sigmoid)

    def forward(self, X):
        """Forward pass through the network."""
        self.input_layer = X
        self.hidden_layer_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_layer_output = self.activation(self.hidden_layer_input)
        self.output_layer_input = np.dot(self.hidden_layer_output, self.weights_hidden_output) + self.bias_output
        self.output_layer_output = self.output_layer_input  # Linear output
        return self.output_layer_output

    def backward(self, X, y):
        """Backward pass to compute gradients and update weights."""
        # Gradient of loss
        loss_gradient = 2 * (self.output_layer_output - y) / y.size

        # Gradients for output layer
        grad_weights_hidden_output = np.dot(self.hidden_layer_output.T, loss_gradient)
        grad_bias_output = np.sum(loss_gradient, axis=0, keepdims=True)

        # Gradients for hidden layer
        grad_hidden_output = np.dot(loss_gradient, self.weights_hidden_output.T)
        grad_hidden_input = grad_hidden_output * self.activation_derivative(self.hidden_layer_input)
        grad_weights_input_hidden = np.dot(self.input_layer.T, grad_hidden_input)
        grad_bias_hidden = np.sum(grad_hidden_input, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= self.lr * grad_weights_hidden_output
        self.bias_output -= self.lr * grad_bias_output
        self.weights_input_hidden -= self.lr * grad_weights_input_hidden
        self.bias_hidden -= self.lr * grad_bias_hidden

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int) * 2 - 1  # Circular boundary
    y = y.reshape(-1, 1)
    return X, y

def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)

    # Hidden features for visualization
    hidden_features = mlp.hidden_layer_output

    # Plot hidden features in 3D
    if hidden_features.shape[1] == 3:
        ax_hidden.scatter(
            hidden_features[:, 0],
            hidden_features[:, 1],
            hidden_features[:, 2],
            c=y.ravel(),
            cmap='bwr',
            alpha=0.7
        )

    # Input space decision boundary
    ax_input.set_title(f"Input Space at Frame {frame * 10}")
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100),
        np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 100)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = mlp.forward(grid).reshape(xx.shape)
    ax_input.contourf(xx, yy, preds, levels=[-1, 0, 1], cmap='bwr', alpha=0.5)
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolor='k')

    # Simple visualization of network structure
    layer_nodes = [2, 3, 1]
    positions = []
    for i, nodes in enumerate(layer_nodes):
        x = np.linspace(0.2, 0.8, nodes)
        y = np.full(nodes, 1 - i * 0.5)
        positions.append(list(zip(x, y)))

    for layer in positions:
        for node in layer:
            ax_gradient.scatter(*node, s=200, color='blue')

    for i, (layer1, layer2) in enumerate(zip(positions[:-1], positions[1:])):
        for j, start in enumerate(layer1):
            for k, end in enumerate(layer2):
                grad = abs(mlp.weights_input_hidden[j, k]) if i == 0 else abs(mlp.weights_hidden_output[j, k])
                ax_gradient.plot(
                    [start[0], end[0]], [start[1], end[1]],
                    color='purple', linewidth=1 + grad * 5, alpha=0.7
                )

    ax_gradient.set_xlim(0, 1)
    ax_gradient.set_ylim(0, 1)
    ax_gradient.axis("off")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    ani = FuncAnimation(
        fig,
        partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y),
        frames=step_num // 10,
        repeat=False
    )

    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
