import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#call mnist function

learningRate = 1e-4
trainingIters = 2200
batchSize = 50
displayStep = 200

nInput = 28#we want the input to take the 28 pixels
nSteps = 28#every 28
nHidden = 128#number of neurons for the RNN
nClasses = 10#this is MNIST so you know

x = tf.placeholder('float', [None, nSteps, nInput])
y = tf.placeholder('float', [None, nClasses])

weights = {
	'out': tf.Variable(tf.random_normal([nHidden, nClasses]))
}

biases = {
	'out': tf.Variable(tf.random_normal([nClasses]))
}


def RNN(x, weights, biases):
    x =tf.unstack(x, nSteps, 1)
    lstmCell = rnn_cell.BasicRNNCell(nHidden)#find which lstm to use in the documentation

    outputs, states = rnn.static_rnn(lstmCell, x, dtype= tf.float32)#for the rnn where to get the output and hidden state

    return tf.matmul(outputs[-1], weights['out'])+ biases['out']


pred = RNN(x, weights, biases)

pred1 = tf.nn.softmax(pred)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate= learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for step in range(9900):
        batchX, batchY = mnist.train.next_batch(batchSize)
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY})

        if step % 100 == 0:
            loss = sess.run(cost, feed_dict= {x: batchX, y:batchY})
            acc = sess.run(accuracy, feed_dict= {x: batchX, y:batchY})
            print("step %d, training accuracy %g" % (step, acc))


    print('Optimisation finished')

    testData = mnist.test.images.reshape((-1,nSteps,nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", \
          sess.run(accuracy,feed_dict= {x: testData, y:testLabel}))
