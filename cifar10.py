from scipy import misc
import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib as mp
from skimage import color


# --------------------------------------------------
# setup

def weight_variable(shape):
    '''
    Initialize weights
    :param shape: shape of weights, e.g. [w, h ,Cin, Cout] where
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters
    Cout: the number of filters
    :return: a tensor variable for weights with initial values
    '''

    # IMPLEMENT YOUR WEIGHT_VARIABLE HERE
    initial = tf.truncated_normal(shape, stddev=0.05)

    return tf.Variable(initial)


def bias_variable(shape):
    '''
    Initialize biases
    :param shape: shape of biases, e.g. [Cout] where
    Cout: the number of filters
    :return: a tensor variable for biases with initial values
    '''

    # IMPLEMENT YOUR BIAS_VARIABLE HERE
    initial = tf.constant(0.0, shape=shape)

    return tf.Variable(initial)


def conv2d(x, W):
    """
    Perform 2-D convolution
    :param x: input tensor of size [N, W, H, Cin] where
    N: the number of images
    W: width of images
    H: height of images
    Cin: the number of channels of images
    :param W: weight tensor [w, h, Cin, Cout]
    w: width of the filters
    h: height of the filters
    Cin: the number of the channels of the filters = the number of channels of images
    Cout: the number of filters
    :return: a tensor of features extracted by the filters, a.k.a. the results after convolution
    """

    # IMPLEMENT YOUR CONV2D HERE
    h_conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    return h_conv


def max_pool_2x2(x):
    '''
    Perform non-overlapping 2-D maxpooling on 2x2 regions in the input data
    :param x: input data
    :return: the results of maxpooling (max-marginalized + downsampling)
    '''

    # IMPLEMENT YOUR MAX_POOL_2X2 HERE
    h_max= tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides= [1, 2, 2, 1], padding= 'SAME')

    return h_max


def colorize(image1, hue, saturation = 1):
    hsv = color.rgb2hsv(image1)
    hsv[:,:,1] = saturation
    hsv[:,:,0] = hue
    rgb1 = color.hsv2rgb(hsv)
    return rgb1


def main():

    result_dir = './results/'
    n_train = 1000      # per class
    n_test = 100        # per class
    nclass = 10         # number of classes
    imsize = 28
    nchannels = 3
    batchsize = 80
    nsamples = 1000

    Train = np.zeros((n_train * nclass, imsize, imsize, nchannels))
    Test = np.zeros((n_test * nclass, imsize, imsize, nchannels))
    LTrain = np.zeros((n_train * nclass, nclass))
    LTest = np.zeros((n_test * nclass, nclass))

    itrain = -1
    itest = -1
    for iclass in range(0, nclass):
        for isample in range(0, n_train):
            path = '/home/chetan/PycharmProjects/Deep-Learning/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
            im = misc.imread(path);  # 28 by 28
            im = color.gray2rgb(im)
            ran2 = np.random.randint(6)
            im1 = colorize(im, ran2, saturation=0.3)
            ran3 = np.random.randint(2)

            # 28 by 28
            im = im.astype(float) / 255
            im1 = im1.astype(float) / 255
            ran1 = np.random.randint(2)

            itrain += 1
            if ran1 == 0:
                if ran3 == 0:
                    Train[itrain, :, :] = im
                else:
                    Train[itrain, :, :] = np.fliplr(im)
            else:
                if ran3 == 0:
                    Train[itrain, :, :] = im1
                else:
                    Train[itrain, :, :] = np.fliplr(im1)
            LTrain[itrain, iclass] = 1  # 1-hot lable
        for isample in range(0, n_test):
            path = '/home/chetan/PycharmProjects/Deep-Learning/CIFAR10/Train/%d/Image%05d.png' % (iclass,isample)
            im = misc.imread(path); # 28 by 28
            im = im.astype(float)/255
            im = color.gray2rgb(im)
            itest += 1
            Test[itest,:,:] = im
            LTest[itest,iclass] = 1 # 1-hot lable

    sess = tf.InteractiveSession()

    tf_data = tf.placeholder(tf.float32, shape= [None, imsize, imsize, nchannels])#tf variable for the data, remember shape is [None, width, height, numberOfChannels]
    tf_labels = tf.placeholder(tf.float32, shape= [None, nclass])#tf variable for labels

    # --------------------------------------------------
    # model
    #create your model
    #w_conv1 = weight_variable([5,5,1,32])
    w_conv1 = tf.get_variable("W1", shape=[5,5,3,32],
                    initializer=tf.contrib.layers.xavier_initializer())
    b_conv1 = bias_variable([32])

    h_conv1 = tf.nn.relu(conv2d(tf_data,w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    #w_conv2 = weight_variable([5,5,32,64])
    w_conv2 = tf.get_variable("W2", shape=[5, 5, 32, 64],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_conv2 = bias_variable([64])

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2,[-1, 7*7*64])

    #W_fc1 = weight_variable([7*7*64, 1024])
    W_fc1 = tf.get_variable("Wfc1", shape=[7*7*64, 1024],
                              initializer=tf.contrib.layers.xavier_initializer())
    b_fc1 = bias_variable([1024])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    #W_fc2 = weight_variable([1024, 10])
    W_fc2 = tf.get_variable("Wfc2", shape=[1024,10],
                            initializer=tf.contrib.layers.xavier_initializer())

    b_fc2 = bias_variable([10])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    # --------------------------------------------------
    # loss
    #set up the loss, optimization, evaluation, and accuracy
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_labels, logits=h_fc2))
    optimizer = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(h_fc2, 1), tf.argmax(tf_labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # --------------------------------------------------
    # optimization
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar(cross_entropy.op.name, cross_entropy)
    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.summary.merge_all()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Instantiate a SummaryWriter to output summaries and the Graph.
    summary_writer = tf.summary.FileWriter(result_dir, sess.graph)


    sess.run(tf.initialize_all_variables())
    batch_xs = np.zeros([batchsize,imsize,imsize,nchannels])#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    batch_ys = np.zeros([batchsize,nclass])#setup as [batchsize, the how many classes]

    for i in range(200): # try a small iteration size once it works then continue
        perm = np.arange(nsamples)
        np.random.shuffle(perm)
        for j in range(batchsize):
            batch_xs[j,:,:,:] = Train[perm[j],:,:,:]
            batch_ys[j,:] = LTrain[perm[j],:]
        if i%100 == 0:
            train_acc = accuracy.eval(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 1.0})
            print("train accuracy %g"%train_acc) #calculate train accuracy and print it
            cross_entro = cross_entropy.eval(feed_dict={tf_data: batch_xs,
                                                        tf_labels: batch_ys, keep_prob: 1.0})
            print("step %d, cross entropy %g" % (i, cross_entro))

            summary_str = sess.run(summary_op, feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})
            summary_writer.add_summary(summary_str, i)
            summary_writer.flush()
        optimizer.run(feed_dict={tf_data: batch_xs, tf_labels: batch_ys, keep_prob: 0.5})

        # dropout only during training

    # --------------------------------------------------
    # test

    print("test accuracy %g"%accuracy.eval(feed_dict={tf_data: Test, tf_labels: LTest, keep_prob: 1.0}))

    sess.close()

if __name__ == "__main__":
    main()