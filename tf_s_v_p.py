import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

### Session ###
# create two matrixes

matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1,matrix2)

# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)


### Variable ###
current = tf.Variable(0, name='counter')
one = tf.constant(1) # 常量

# Define the addition step
new_value = tf.add(current, one)

# update State into new_value
update = tf.assign(current, new_value)

# must initialize if variable has defined
init = tf.global_variables_initializer()  
 
with tf.Session() as sess:
    sess.run(init)
    for _ in range(5):
        sess.run(update)
        print(sess.run(current))



### Placeholder ###
#define the type of placeholder, generally in the form of float32
input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

ouput = tf.multiply(input1, input2)

# The value that needs to be passed in is placed in feed_dict=() and corresponds to each input one by one.
with tf.Session() as sess:
    print(sess.run(ouput, feed_dict={input1: [7.], input2: [2.]}))
# [ 14.]