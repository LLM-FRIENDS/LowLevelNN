## Forward Pass

The input data $X$ is propagated through the network to produce the output $A_2$:

1. **Hidden Layer**:
   
   $$
   Z_1 = W_1 X + b_1
   $$
   
   $$
   A_1 = \sigma(Z_1)
   $$

2. **Output Layer**:
   
   $$
   Z_2 = W_2 A_1 + b_2
   $$
   
   $$
   A_2 = \sigma(Z_2)
   $$

## Loss Calculation

The Mean Squared Error (MSE) is used as the loss function:

$$
\text{Loss} = \frac{1}{m} \sum_{i=1}^{m} (A_2^{(i)} - y^{(i)})^2
$$

For the dummy case ($m=1$):

$$
\text{Loss} = (A_2 - y)^2
$$

## Chain Rule

The chain rule is a fundamental principle in calculus that allows us to compute the derivative of a composite function by breaking it down into simpler parts. 

**Example:**  
Consider a function that transforms an input $x$ to an intermediate variable $z$, and then to an output $y$:
$$
x \rightarrow z \rightarrow y
$$

To compute the derivative of $y$ with respect to $x$ ($\frac{\partial y}{\partial x}$), we apply the chain rule:
$$
\frac{\partial y}{\partial x} = \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial x}
$$

In the context of neural networks, this principle allows us to calculate the gradients of the loss function with respect to each parameter by systematically applying partial derivatives through each layer of the network. Mathematically, if a loss function $L$ depends on an intermediate variable $A$, which in turn depends on $Z$, which depends on $W$, the chain rule allows us to express the derivative of $L$ with respect to $W$ as:
$$
\frac{\partial L}{\partial W} = \frac{\partial L}{\partial A} \cdot \frac{\partial A}{\partial Z} \cdot \frac{\partial Z}{\partial W}
$$
This decomposition simplifies the computation of complex derivatives, making backpropagation feasible and efficient.

## Backward Pass

The backward pass involves computing the gradients of the loss with respect to each parameter in the network using the chain rule.

### Step 1: Compute $\frac{\partial \text{Loss}}{\partial A_2}$

Given the loss function for the dummy case:

$$
\text{Loss} = (A_2 - y)^2
$$

The derivative with respect to $A_2$ is:

$$
\frac{\partial \text{Loss}}{\partial A_2} = 2(A_2 - y)
$$

### Step 2: Compute $\frac{\partial A_2}{\partial Z_2}$

Since $A_2 = \sigma(Z_2)$, where $\sigma$ is the sigmoid activation function:

$$
\sigma(Z) = \frac{1}{1 + e^{-Z}}
$$

The derivative of the sigmoid function with respect to its input is:

$$
\frac{\partial \sigma(Z)}{\partial Z} = \sigma(Z)(1 - \sigma(Z))
$$

Therefore:

$$
\frac{\partial A_2}{\partial Z_2} = A_2 (1 - A_2)
$$

### Step 3: Compute $\frac{\partial \text{Loss}}{\partial Z_2}$

Using the chain rule:

$$
\frac{\partial \text{Loss}}{\partial Z_2} = \frac{\partial \text{Loss}}{\partial A_2} \cdot \frac{\partial A_2}{\partial Z_2} = 2(A_2 - y) \cdot A_2(1 - A_2)
$$

Simplifying:

$$
\delta_2 = (A_2 - y) \cdot A_2(1 - A_2)
$$

### Step 4: Compute Gradients for $W_2$ and $b_2$

The gradients with respect to $W_2$ and $b_2$ are:

$$
\frac{\partial \text{Loss}}{\partial W_2} = \delta_2 \cdot A_1^T
$$

$$
\frac{\partial \text{Loss}}{\partial b_2} = \delta_2
$$

### Step 5: Compute $\frac{\partial \text{Loss}}{\partial A_1}$

Since $Z_2 = W_2 A_1 + b_2$, the derivative of the loss with respect to $A_1$ is:

$$
\frac{\partial \text{Loss}}{\partial A_1} = W_2^T \cdot \delta_2
$$

### Step 6: Compute $\frac{\partial A_1}{\partial Z_1}$

Similarly to the output layer:

$$
\frac{\partial A_1}{\partial Z_1} = A_1(1 - A_1)
$$

### Step 7: Compute $\frac{\partial \text{Loss}}{\partial Z_1}$

Using the chain rule:

$$
\frac{\partial \text{Loss}}{\partial Z_1} = \frac{\partial \text{Loss}}{\partial A_1} \cdot \frac{\partial A_1}{\partial Z_1} = W_2^T \cdot \delta_2 \cdot A_1(1 - A_1)
$$

Simplifying:

$$
\delta_1 = (W_2^T \cdot \delta_2) \cdot A_1(1 - A_1)
$$

### Step 8: Compute Gradients for $W_1$ and $b_1$

The gradients with respect to $W_1$ and $b_1$ are:

$$
\frac{\partial \text{Loss}}{\partial W_1} = \delta_1 \cdot X^T
$$

$$
\frac{\partial \text{Loss}}{\partial b_1} = \delta_1
$$

## Update Parameters

Using the computed gradients, update the weights and biases:

$$
W_2 \leftarrow W_2 - \eta \frac{\partial \text{Loss}}{\partial W_2}
$$

$$
b_2 \leftarrow b_2 - \eta \frac{\partial \text{Loss}}{\partial b_2}
$$

$$
W_1 \leftarrow W_1 - \eta \frac{\partial \text{Loss}}{\partial W_1}
$$

$$
b_1 \leftarrow b_1 - \eta \frac{\partial \text{Loss}}{\partial b_1}
$$

where $\eta$ is the learning rate.

## Training Process

1. **Forward Pass**: Compute the output $A_2$.
2. **Compute Loss**: Calculate MSE between $A_2$ and true label $y$.
3. **Backward Pass**: Perform backpropagation to compute gradients.
4. **Update Parameters**: Adjust $W_1$, $b_1$, $W_2$, and $b_2$ to minimize the loss.

Monitoring the loss each epoch helps track the network's performance and guide the optimization process.
