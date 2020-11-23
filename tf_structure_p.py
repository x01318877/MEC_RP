# let tensorflow to learn that make the value of Weights trend to 0.1 and biases trend to 0.3

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# create data #
x_data = np.random.rand(100).astype(np.float32)
# the result we want to achieve
y_data = x_data*0.1 + 0.3

# Build Model #
### create tensorflow structure start ###
# parameters: Weights, [1]-D, initial value range(-1~1)
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
# parameters: biases, [1]-D, initial value 0
biases = tf.Variable(tf.zeros([1]))

y = Weights*x_data + biases

# Calculate the error of y and y_data #
loss = tf.reduce_mean(tf.square(y-y_data))
# optimizer: Error transfer (Gradient Descent), and then use optimizer to update the parameters #
# 0.5 means 学习效率，always below 1
optimizer = tf.train.GradientDescentOptimizer(0.5)
# the key idea of NN is to min the value of loss
train = optimizer.minimize(loss)
### create tensorflow structure end ###

# Initialize all previously defined Variable #
init = tf.global_variables_initializer()

# Use Session to perform the init initialization step #
# Use Session to run the data of each training. Gradually improve the prediction accuracy of the neural network #
sess = tf.Session()
sess.run(init)

# train 201 step
for step in range(201):
    sess.run(train)
    # Print the result every 20 steps, sess.run points to Weights, biases and show the output
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))