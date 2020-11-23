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
</p>

In this image, W's Global minima is in this position(orange), and the other solutions are Local minima. The global optimal is the best, but in many cases, you have A local optimal solution. The neural network can also make your local optimal good enough that even if you hold a local optimal, you can perform the task in your hand well.


