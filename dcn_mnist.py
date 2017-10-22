import os
import time
import tensorflow as tf
from scipy import misc
import numpy as np
import random
#import matplotlib.pyplot as plt
#import matplotlib as mp
from skimage import color


# Import Tensorflow and start a session

def weight_variable(shape):

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape,stddev=0.1)

    return tf.Variable(initial)


def bias_variable(shape):

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.1,shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):

    return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')


def hist(x,str1):
    tf.summary.histogram(str1, x)
    tf.summary.scalar(str1 + 'min', tf.reduce_min(x))
    tf.summary.scalar(str1 + 'max', tf.reduce_max(x))
    tf.summary.scalar(str1 + 'mean', tf.reduce_mean(x))
    mean = tf.reduce_mean(x)
    tf.summary.scalar(str1 + 'sd',tf.sqrt(tf.reduce_mean(tf.square(x - mean))))


def max_pool_2x2(x):

    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')



# Specify training parameters
result_dir = './results/' # directory where the results from the training are saved
max_step = 5500 # the maximum iterations. After max_step iterations, the training will stop no matter what

start_time = time.time() # start timing
n_train = 1000  # per class
n_test = 100  # per class
nclass = 10  # number of classes
imsize = 28
nchannels = 1
batchsize = 100
nsamples = 10000

Train = np.zeros((n_train * nclass, imsize, imsize, nchannels))
Test = np.zeros((n_test * nclass, imsize, imsize, nchannels))
LTrain = np.zeros((n_train * nclass, nclass))
LTest = np.zeros((n_test * nclass, nclass))

itrain = -1
itest = -1
for iclass in range(0, nclass):
    for isample in range(0, n_train):
        path = '~/Deep-Learning/CIFAR10/Train/%d/Image%05d.png' % (iclass, isample)
        path = os.path.expanduser(path)
        im = misc.imread(path);  # 28 by 28

        # 28 by 28
        im = im.astype(float) / 255

        itrain += 1
        Train[itrain, :, :,0] = im
        LTrain[itrain, iclass] = 1

    for isample in range(0, n_test):
        path = '~/Deep-Learning/CIFAR10/Test/%d/Image%05d.png' % (iclass, isample)
        path = os.path.expanduser(path)
        im = misc.imread(path);
        im = im.astype(float) / 255
        itest += 1
        Test[itest, :, :,0] = im
        LTest[itest, iclass] = 1  # 1-hot lable

# FILL IN THE CODE BELOW TO BUILD YOUR NETWORK
sess = tf.InteractiveSession()
# placeholders for input data and input labeles
x = tf.placeholder(tf.float32, shape=[None, imsize, imsize, nchannels])

y_ = tf.placeholder(tf.float32, shape=[None,10])



# first convolutional layer


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# second convolutional layer
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# densely connected layer

W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
#W_fc2 = tf.get_variable("W_fc2", shape=[1024,10],
#                        initializer=tf.contrib.layers.xavier_initializer())
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# FILL IN THE FOLLOWING CODE TO SET UP THE TRAINING
W1_a = W_conv1  # [5, 5, 1, 32]
W1pad = tf.zeros([5, 5, 1, 1])  # [5, 5, 1, 4]  - four zero kernels for padding
# We have a 6 by 6 grid of kernepl visualizations. yet we only have 32 filters
# Therefore, we concatenate 4 empty filters
W1_b = tf.concat(3, [W1_a, W1pad, W1pad, W1pad, W1pad])  # [5, 5, 1, 36]
W1_c = tf.split(3, 36, W1_b)  # 36 x [5, 5, 1, 1]
W1_row0 = tf.concat(0, W1_c[0:6])  # [30, 5, 1, 1]
W1_row1 = tf.concat(0, W1_c[6:12])  # [30, 5, 1, 1]
W1_row2 = tf.concat(0, W1_c[12:18])  # [30, 5, 1, 1]
W1_row3 = tf.concat(0, W1_c[18:24])  # [30, 5, 1, 1]
W1_row4 = tf.concat(0, W1_c[24:30])  # [30, 5, 1, 1]
W1_row5 = tf.concat(0, W1_c[30:36])  # [30, 5, 1, 1]
W1_d = tf.concat(1, [W1_row0, W1_row1, W1_row2, W1_row3, W1_row4, W1_row5])  # [30, 30, 1, 1]
W1_e = tf.reshape(W1_d, [1, 30, 30, 1])
Wtag = tf.placeholder(tf.string, None)
image_summary_t = tf.image_summary("Visualize_kernels", W1_e)

# setup training
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits= y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.to_float(correct_prediction))

# hist(x_image,'x_img')

test_sum = tf.summary.scalar("test accuracy",accuracy)
# Add a scalar summary for the snapshot loss.
# Build the summary operation based on the TF collection of Summaries.
summary_op = tf.summary.merge_all()

# Add the variable initializer Op.
init = tf.initialize_all_variables()

# Create a saver for writing training checkpoints.
saver = tf.train.Saver()

# Instantiate a SummaryWriter to output summaries and the Graph.
summary_writer = tf.summary.FileWriter(result_dir, sess.graph)

# Run the Op to initialize the variables.
sess.run(init)

batch_xs = np.zeros([batchsize, imsize, imsize, nchannels])  # setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
batch_ys = np.zeros([batchsize, nclass])  # setup as [batchsize, the how many classes]

# run the training
perm = np.arange(nsamples)
for i in range(5500):

    np.random.shuffle(perm)
    for j in range(batchsize):
        batch_xs[j, :, :, :] = Train[perm[j], :, :, :]
        batch_ys[j, :] = LTrain[perm[j], :] # make the data batch, which is used in the training iteration.
                                        # the batch size is 50
    if i % 100 == 0:
        # output the training accuracy every 100 iterations

        train_accuracy = accuracy.eval(feed_dict={x: batch_xs,
                                                  y_: batch_ys, keep_prob: 1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))

        # Update the events file which is used to monitor the training (in this case,
        # only the training loss is monitored)
        summary_str = sess.run(summary_op, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
        summary_writer.add_summary(summary_str, i)
        summary_writer.flush()

    # save the checkpoints every 1100 iterations
    if i % 1100 == 0 or i == max_step:
        test_accuracy, test_summ = sess.run([accuracy, test_sum],
                                            feed_dict={x: Test,
                                                       y_: LTest,
                                                       keep_prob: 1.0})
        summary_writer.add_summary(test_summ, i)
        print("test: step %d, accuracy %g" % (i, test_accuracy))
        checkpoint_file = os.path.join(result_dir, 'checkpoint')
        saver.save(sess, checkpoint_file, global_step=i)

    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5}) # run one train_step

# print test error
print("test accuracy %g"%accuracy.eval(feed_dict={x: Test,
                                                  y_: LTest, keep_prob: 1.0}))

stop_time = time.time()
print('The training takes %f second to finish'%(stop_time - start_time))

