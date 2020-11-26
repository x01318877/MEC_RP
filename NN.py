import tensorflow as tf
import numpy as np

# add_layer
def add_layer(inputs, in_size, out_size, activation_function=None):
   # add one more layer and return the output of this layer
   Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
   Wx_plus_b = tf.matmul(inputs, Weights) + biases
   if activation_function is None:
       outputs = Wx_plus_b
   else:
       outputs = activation_function(Wx_plus_b)
   return outputs

# 1.data that need to be trained
# 300 data, between (-1 ~ 1)
x_data = np.linspace(-1,1,300)[:, np.newaxis]
# Make up some real data , distribute data point
noise = np.random.normal(0, 0.05, x_data.shape)
# ^2 square
y_data = np.square(x_data) - 0.5 + noise

# 2.define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

# 3.Define the neural layer: hidden layer and prediction layer
# add hidden layer input value xs，10 neurons in the hidden layer   
l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
# add output layer, The input value is the hidden layer l1, and 1 result is output in the prediction layer
prediction = add_layer(l1, 10, 1, activation_function=None)

# 4.Define loss expression
# the error between prediciton and real data    
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction),
                    reduction_indices=[1]))

# 5.Choose optimizer min loss                   
# This line defines how to reduce loss, the learning rate is 0.1      
train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


# important step - initialize
init = tf.initialize_all_variables()
sess = tf.Session()

sess.run(init)

# 1000 iterations of learning，sess.run optimizer
for i in range(1000):
   sess.run(train_step, feed_dict={xs: x_data, ys: y_data})
   if i % 50 == 0:
       # to see the step improvement
       print(sess.run(loss, feed_dict={xs: x_data, ys: y_data}))
