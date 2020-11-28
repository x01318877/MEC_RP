# Tensorflow
https://www.jianshu.com/p/e112012a4b2d
- TensorFlow allows us to draw a calculation structure diagram first, or it can be called a series of human-computer interactive calculation operations, and then convert the edited Python file into a more efficient C++, and perform calculations on the backend.

- The task it is good at is training deep neural networks.

- By using TensorFlow, we can quickly get started with neural networks, which greatly reduces the development cost and difficulty of deep learning (that is, deep neural networks)

### Neural Network
1. input layer -> hidden layer -> output layer
2. Activation function: After training again and again, the parameters of all neurons are being changed, and they become more sensitive to really important information

### Gradient Descent
<p>
<img src="https://user-images.githubusercontent.com/23052423/99918204-4f1c1f80-2d0d-11eb-966d-0d5cdfb387b6.png" width="400" height="180">

When the cost error is the smallest, it is the lowest point of the cost curve, but W at the blue dot does not know this. What he currently knows is that the gradient line points to a downward direction for himself at this position. Go down a little bit in the direction of this blue gradient. After making a tangent, I found that I can still fall, then I will continue to decline in the direction of the gradient. At this time, show the gradient that appears because the gradient line is already flat Now, we can't figure out which side is the direction of descent, so at this time we have found the optimal value of the W parameter. In short, we find the point where the gradient line lies flat.
  
<img src="https://user-images.githubusercontent.com/23052423/99918206-504d4c80-2d0d-11eb-85f1-f1fed8d98c94.png" width="400" height="180">

In this image, W's Global minima is in this position(orange), and the other solutions are Local minima. The global optimal is the best, but in many cases, you have A local optimal solution. The neural network can also make your local optimal good enough that even if you hold a local optimal, you can perform the task in your hand well.

## Regression

### 1. Tensorflow Processing Structure

<img src="https://user-images.githubusercontent.com/23052423/99979340-e4b3bf80-2d9e-11eb-8c1e-1cc8c7e28251.gif" width="300" height="500">

The explanation of this animation is that the data -> input layer -> hidden layer -> output layer. It is processed by gradient descent. Gradient descent will update and improve several parameters, the updated parameters will run to the hidden layer and learn again, until the result converges.

Tensor: 
- The zero-order tensor is a scalar or a scalar, which is a value. For example, [1] 
- The first-order tensor is a vector, such as one-dimensional [ 1, 2, 3] 
- The second-order tensor is a matrix, such as a two-dimensional [[1, 2, 3],[4, 5, 6],[7, 8, 9]] 


### 2. Tensorflow - Session & Variable & Placeholder
(1) Session

    sess = tf.Session()

(2) Variable
    
   In Tensorflow, a string must be defined as a variable, then it is a variable, which is different from Python.

    state = tf.Variable()

(3) Placeholder
- Use placeholder to describe the node waiting for input, just specify the type

- It is equivalent to hold the variable first, and then pass in data from the outside each time. Note that placeholder and feed_dict are used for binding.

- The feed is only valid in the method that calls it, and the method ends, the feed will disappear

- feed_dict -> dictionary form

        tf.placeholder()    
        sess.run(***, feed_dict={input: **}).

### 3. Activation Function
(1) nonlinear function

In a small number of layer structures, we can try many different excitation functions. 

- In the convolutional layer of Convolutional neural networks, the recommended excitation function is relu. 

- In recurrent neural networks, tanh or relu is recommended 

(2) Process

- layer1 and layer2 are both hidden layers, pass the value to the predictor and calculate the cost between predicte value and fact value.

<img src="https://user-images.githubusercontent.com/23052423/100358452-1ff7fd80-2fee-11eb-957a-795bec123326.png" width="300" height="230">

- This is extension of layer2. layer1 pass the value to layer2, layer2 process the value and see what value need to be avtived then prediction

<img src="https://user-images.githubusercontent.com/23052423/100358460-21c1c100-2fee-11eb-9d5c-71e25d3eca47.png" width="300" height="230">
</p>

(3) Functions example

https://www.tensorflow.org/api_docs/python/tf/keras/activations

relu(...): Applies the rectified linear unit activation function.

selu(...): Scaled Exponential Linear Unit (SELU).

serialize(...): Returns the string identifier of an activation function.

sigmoid(...): Sigmoid activation function, sigmoid(x) = 1 / (1 + exp(-x)).

softmax(...): Softmax converts a real vector to a vector of categorical probabilities.

softplus(...): Softplus activation function, softplus(x) = log(exp(x) + 1).

softsign(...): Softsign activation function, softsign(x) = x / (abs(x) + 1).

swish(...): Swish activation function, swish(x) = x * sigmoid(x).

tanh(...): Hyperbolic tangent activation function.



### 4. add_layer()

https://mofanpy.com/tutorials/machine-learning/tensorflow/add-layer/

Defining a function to add layers in Tensorflow can easily add neural layers

- Neural layer Parameter: weights、biases and Activation Function

(1) Function def add_layer() has four parameters: input value, input size, output size and Activation Function. We set the default Activation Function to be None

     def add_layer(inputs, in_size, out_size, activation_function=None):    

(2) Define weights and biases

When generating the initial parameters, the normal distribution will be much better than all 0, so our weights here is a random variable matrix with in_size rows and out_size columns.

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    
In machine learning, the recommended value of biases is not 0, so here we add 0.1 to the 0 vector.

    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    
Below, we define Wx_plus_b, which is the value of the inactive neural network. Among them, tf.matmul() is matrix multiplication.

    Wx_plus_b = tf.matmul(inputs, Weights) + biases

When activation_function-the activation function is None, the output is the current predicted value-Wx_plus_b, when it is not None, pass Wx_plus_b to the activation_function() function to get the output.

    if activation_function is None:
         outputs = Wx_plus_b
     else:
         outputs = activation_function(Wx_plus_b)

Finally, return the output and add a neural layer function-def add_layer() is defined.

    return outputs


### 5. Building a neural network + matplotlib
    NN.py
    
### 6. TensorBoard
    tensorboard.py

### 7. Overfitting - dropout regulation

