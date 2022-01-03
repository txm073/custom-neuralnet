import numpy as np

# Rectified linear units activation function
# Element-wise formula: 
#   if element > 0: -> element
#   else if element <= 0: -> 0
def relu(x):
    return np.maximum(0, x)

# Softmax activation function
# Formula:
#   exponentiation to remove all negative numbers in the vector
#   normalisation by dividing each neuron's output by the sum of all the outputs in the layer
#   returns a probability distribution for each batch (a.k.a each vector in the matrix)
def softmax(x):
    matrix = []
    for vector in x:
        exp_values = np.exp(vector)
        norm_values = exp_values / np.sum(exp_values)
        matrix.append(norm_values)
    return np.array(matrix)

# Compute loss with the categorical cross entropy function
# Formula:
#   the network's predicted values are inputted after the softmax function to score each class
#   the real values are one-hot encoded so that the x and y vectors are the same length
#   iterate over both vectors simultaneously and create a new 'log list'
#   append the real element (a 1 or 0) * by the natural log of the predicted element to the new list
#   calculate the negative sum of the log list
#   repeat the process for each output in the data batch 
def cce(x, y):
    loss_matrix = []
    for pred, real in zip(x, y):
        loss = -1 * sum([np.log(pred_elem) * real_elem for pred_elem, real_elem in zip(pred, real)])
        loss_matrix.append(loss)
    return np.mean(loss_matrix)

def sparse_cce(x, y):
    # 2D iteration across Numpy array
    return -np.log(x[range(len(x)), y])    

# One-hot encode the target values (labels) to be a vector of 0s, of length n
# Where n is equal to the number of classes
# The element of the index of the target class will be replaced with a 1
def one_hot_encode(y, n):
    vector = [0] * n
    vector[y] = 1
    return vector

# Base class for all layers to enforce correct parameters in the model class
class Layer: ...

# Fully connected layer 
class Dense(Layer):

    def __init__(self, inputs, neurons, activation):
        # Initialise random weights
        self.weights = np.random.randn(inputs, neurons) * 0.1
        # Initialise the bias for each neuron as 0
        self.biases = np.zeros((1, neurons))
        self.activation = activation
    
    def __call__(self, x):
        return self.activation(np.dot(x, self.weights) + self.biases)    

# Class for traditional feed-forward neural network
# Data is passed through each layer in sequence
class Sequential:

    def __init__(self, layers=None):
        self.layers = layers
        if self.layers is not None:
            for i, l in enumerate(self.layers):
                assert isinstance(l, Layer), f"Object at index {i} is not a layer!"
        else:
            self.layers = []

    def __call__(self, x):
        self.layers_passed = 0
        self.complete = False
        self._forward(x)
        return self._output

    # Recursively feed the data batch through each layer in the network
    # Assume that the network has less than 1000 layers to avoid a recursion limit
    def _forward(self, x):
        if self.layers_passed == len(self.layers) - 1:
            self.complete = True

        self._output = self.layers[self.layers_passed](x)
        if self.complete is False:
            self.layers_passed += 1
            self._forward(self._output)
        else:
            return 

    def evaluate():
        pass

    def fit(x, y):
        pass

    def compile(optimiser=None, loss=cce):
        pass

    def add(self, layer):
        assert isinstance(layer, Layer), "Object passed is not a layer!"
        self.layers.append(layer)

if __name__ == "__main__":
    classes = 5

    x = [[1.0, 0.8, 2.3, 0.6],
         [0.3, 0.7, 1.3, -2.0],
         [-1.5, 2.7, -0.2, 1.7]]
    
    y = [0, 
         3, 
         2]
    
    layer1 = Dense(inputs=4, neurons=10, activation=relu)
    layer2 = Dense(inputs=10, neurons=classes, activation=softmax)

    model = Sequential([layer1, "layer2", layer2])

    output = model(x)
    targets = np.array([one_hot_encode(index, classes) for index in y])    
    
    print(output)
    print("\n\n")

    print(cce(output, targets))
    
