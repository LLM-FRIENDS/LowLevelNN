import numpy as np

class NeuralNetwork:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initializes the neural network with given dimensions.
        
        Parameters:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Number of output neurons.
        """
        # Xavier (Glorot) Initialization for weights with sigmoid activation
        self.W1 = np.random.randn(hidden_dim, input_dim) * np.sqrt(1. / input_dim)
        self.b1 = np.zeros((hidden_dim, 1))  # Bias for hidden layer
        self.W2 = np.random.randn(output_dim, hidden_dim) * np.sqrt(1. / hidden_dim)
        self.b2 = np.zeros((output_dim, 1))  # Bias for output layer

    def sigmoid(self, x):
        """
        Applies the sigmoid activation function.
        
        Parameters:
            x (numpy.ndarray): Input array.
        
        Returns:
            numpy.ndarray: Sigmoid activation applied element-wise.
        """
        # Clip x to prevent overflow in exp
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.
        
        Parameters:
            X (numpy.ndarray): Input data of shape (input_dim, m).
        
        Returns:
            numpy.ndarray: Output of the network after activation.
        """
        # Store input for use in backpropagation
        self.X = X  # Shape: (input_dim, m)

        # Hidden layer computations
        self.Z1 = np.dot(self.W1, X) + self.b1  # Linear combination: Z1 = W1 * X + b1
        self.A1 = self.sigmoid(self.Z1)         # Activation: A1 = sigmoid(Z1)

        # Output layer computations
        self.Z2 = np.dot(self.W2, self.A1) + self.b2  # Linear combination: Z2 = W2 * A1 + b2
        self.A2 = self.sigmoid(self.Z2)               # Activation: A2 = sigmoid(Z2)

        return self.A2  # Shape: (output_dim, m)

    def sigmoid_derivative(self, sigmoid_output):
        """
        Computes the derivative of the sigmoid function.
        
        Parameters:
            sigmoid_output (numpy.ndarray): Output from the sigmoid function.
        
        Returns:
            numpy.ndarray: Derivative of sigmoid.
        """
        return sigmoid_output * (1 - sigmoid_output)

    def backward_propagation(self, y):
        """
        Performs backward propagation to compute gradients.
        
        Parameters:
            y (numpy.ndarray): True labels of shape (output_dim, m).
        """
        m = y.shape[1]  # Number of examples

        # ***** Output Layer Calculations *****
        
        # Compute derivative of loss w.r.t A2 (Output activation)
        # For MSE: dL/dA2 = 2*(A2 - y) / m
        # Simplifying constant 2/m with learning rate adjustment
        dA2 = self.A2 - y  # Shape: (output_dim, m)

        # Compute derivative of loss w.r.t Z2
        # dL/dZ2 = dL/dA2 * dA2/dZ2
        # Since A2 = sigmoid(Z2), dA2/dZ2 = sigmoid_derivative(Z2)
        # Therefore, dZ2 = dA2 * sigmoid_derivative(Z2)
        dZ2 = dA2 * self.sigmoid_derivative(self.A2)  # Shape: (output_dim, m)

        # Compute gradients for W2 and b2
        # dW2 = (1/m) * dZ2 * A1^T
        self.dW2 = (1 / m) * np.dot(dZ2, self.A1.T)  # Shape: (output_dim, hidden_dim)

        # db2 = (1/m) * sum of dZ2 across all examples
        self.db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)  # Shape: (output_dim, 1)

        # ***** Hidden Layer Calculations *****
        
        # Compute derivative of loss w.r.t A1 (Hidden layer activation)
        # dL/dA1 = W2^T * dZ2
        dA1 = np.dot(self.W2.T, dZ2)  # Shape: (hidden_dim, m)

        # Compute derivative of loss w.r.t Z1
        # dL/dZ1 = dL/dA1 * dA1/dZ1
        # Since A1 = sigmoid(Z1), dA1/dZ1 = sigmoid_derivative(Z1)
        # Therefore, dZ1 = dA1 * sigmoid_derivative(Z1)
        dZ1 = dA1 * self.sigmoid_derivative(self.A1)  # Shape: (hidden_dim, m)

        # Compute gradients for W1 and b1
        # dW1 = (1/m) * dZ1 * X^T
        self.dW1 = (1 / m) * np.dot(dZ1, self.X.T)  # Shape: (hidden_dim, input_dim)

        # db1 = (1/m) * sum of dZ1 across all examples
        self.db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)  # Shape: (hidden_dim, 1)

    def update_parameters(self, learning_rate):
        """
        Updates the network's parameters using the computed gradients.
        
        Parameters:
            learning_rate (float): Learning rate for gradient descent.
        """
        # Update weights and biases for output layer
        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

        # Update weights and biases for hidden layer
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

    def train(self, X, y, learning_rate=0.1, epochs=1000):
        """
        Trains the neural network using the provided data.
        
        Parameters:
            X (numpy.ndarray): Input data of shape (input_dim, m).
            y (numpy.ndarray): True labels of shape (output_dim, m).
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training iterations.
        """
        for epoch in range(epochs):
            # ***** Forward Pass *****
            A2 = self.forward_propagation(X)

            # ***** Compute Loss *****
            loss = np.mean((A2 - y) ** 2)  # Mean Squared Error (MSE)

            # ***** Backward Pass *****
            self.backward_propagation(y)

            # ***** Update Parameters *****
            self.update_parameters(learning_rate)

            # ***** Print Loss Every 100 Epochs *****
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}")


    def train_mini_batch(self, X, y, learning_rate=0.1, epochs=1000, batch_size=32):
        """
        Trains the neural network using Mini-Batch Gradient Descent.
        
        Parameters:
            X (numpy.ndarray): Input data of shape (input_dim, m).
            y (numpy.ndarray): True labels of shape (output_dim, m).
            learning_rate (float): Learning rate for gradient descent.
            epochs (int): Number of training iterations.
            batch_size (int): Number of examples per mini-batch.
        """
        # Verify that y has the correct shape
        if y.ndim != 2 or y.shape[0] != 1:
            raise ValueError(f"y should have shape (1, m), but got {y.shape}")
        
        m = X.shape[1]  # Number of training examples

        for epoch in range(epochs):
            # ***** Shuffle the Data *****
            permutation = np.random.permutation(m)
            X_shuffled = X[:, permutation]
            y_shuffled = y[:, permutation]

            # ***** Mini-Batch Processing *****
            for i in range(0, m, batch_size):
                end = i + batch_size
                X_batch = X_shuffled[:, i:end]
                y_batch = y_shuffled[:, i:end]

                # ***** Forward Pass *****
                A2 = self.forward_propagation(X_batch)

                # ***** Compute Loss *****
                loss = np.mean((A2 - y_batch) ** 2)  # MSE for the current batch

                # ***** Backward Pass *****
                self.backward_propagation(y_batch)

                # ***** Update Parameters *****
                self.update_parameters(learning_rate)

            # ***** Compute Overall Loss for the Epoch *****
            A2_full = self.forward_propagation(X)
            loss_full = np.mean((A2_full - y) ** 2)

            # ***** Print Loss Every 100 Epochs *****
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss_full:.6f}")
