import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import os

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)#call mnist function

learningRate = 1e-4
trainingIters = 2200
batchSize = 50
displayStep = 200

result_dir = './results/'

nInput = 28#we want the input to take the 28 pixels
nSteps = 28#every 28
nHidden = 128  #number of neurons for the RNN
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
    x = tf.transpose(x, [1, 0, 2])
    x = tf.reshape(x, [-1, nInput])
    x = tf.split(x, nSteps, 0)

    lstmCell = rnn_cell.BasicRNNCell(nHidden)
    # lstmCell = rnn_cell.BasicLSTMCell(nHidden, forget_bias= 1.0)
    # lstmCell = rnn_cell.GRUCell(nHidden) #find which lstm to use in the documentation

    outputs, states = rnn.static_rnn(lstmCell, x, dtype= tf.float32)#for the rnn where to get the output and hidden state

    return tf.matmul(outputs[-1], weights['out'])+ biases['out']


pred = RNN(x, weights, biases)

pred1 = tf.nn.softmax(pred)

#optimization
#create the cost, optimization, evaluation, and accuracy
#for the cost softmax_cross_entropy_with_logits seems really good
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred,labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate= learningRate).minimize(cost)

correctPred = tf.equal(tf.argmax(pred1, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))
train_sum = tf.summary.scalar("train accuracy",accuracy)

tf.summary.scalar("Cost", cost)
    # Build the summary operation based on the TF collection of Summaries.

summary_op = tf.summary.merge_all()
test_sum = tf.summary.scalar("test accuracy",accuracy)
init = tf.global_variables_initializer()

with tf.Session() as sess:
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

    sess.run(init)

    for step in range(9900):
        batchX, batchY = mnist.train.next_batch(batchSize)
        batchX = batchX.reshape((batchSize, nSteps, nInput))

        sess.run(optimizer, feed_dict={x: batchX, y: batchY})


        if step % 100 == 0:
            loss = sess.run(cost, feed_dict= {x: batchX, y:batchY})
            acc = sess.run(accuracy, feed_dict= {x: batchX, y:batchY})
            print("step %d, training accuracy %g" % (step, acc))
            summary_str = sess.run(summary_op, feed_dict={x: batchX, y: batchY})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()
        if step % 1100 == 0 or step == 9900:
            test_accuracy, test_summ = sess.run([accuracy, test_sum],
                                                feed_dict={x: mnist.test.images.reshape((-1,nSteps,nInput)),
                                                            y: mnist.test.labels})
            summary_writer.add_summary(test_summ, step)
            print("test: step %d, accuracy %g" % (step, test_accuracy))

            checkpoint_file = os.path.join(result_dir, 'checkpoint')
            saver.save(sess, checkpoint_file, global_step=step)
            summary_writer.flush()


    print('Optimisation finished')

    testData = mnist.test.images.reshape((-1,nSteps,nInput))
    testLabel = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy,feed_dict= {x: testData, y:testLabel}))
