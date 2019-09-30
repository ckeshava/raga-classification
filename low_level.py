# Todo:
# Modify the kernel architecture: 
#   - Try out conv3d, input data format [batch, in_depth, in_height, in_width, in_channels]
#   - hyper param tuning



# URGENT
# Error in transporting spectrograms from one file into another!
# this code will break due to changes in lines 17-24.
# Use a 3D CNN kernel.
# trim the input data to fixed dimensions!!
# spitting out accuracy and summary seems to be a pain in the ass

# CRITICAL:
# pooling factors <64 triggers resource constraints. Effect on accuracy??



import tensorflow as tf
import numpy as np
import os, glob
import warnings
warnings.filterwarnings('ignore')

import image_preprocess
import config





# to be modified
def get_data(input_dir="input/"):
    # x = np.random.rand(config.LENGTH, config.BREADTH, config.NUM_CHANNELS)
    # y_true = np.random.rand(config.NUM_RAGAS, 1) 

    x = image_preprocess.get_image(input_dir) 
    y_true = image_preprocess.get_label(input_dir)
    

    return x, y_true  


def cnn_model_fn(x):
    """Model function for CNN."""

    input_layer = tf.reshape(x, [-1, config.LENGTH, config.BREADTH, config.NUM_CHANNELS])

    template = ('DEBUG: FORW PROP: {} shape: {}\n')

    print(template.format(input_layer, input_layer.shape))

    # Convolutional Layer #1
    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        filters=32,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    print(template.format(conv1, conv1.shape))
    

    

    # Pooling Layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[32, 32], strides=32, padding="same")
    # pool1 = conv1

    print(template.format(pool1, pool1.shape))



    # Convolutional Layer #2 and Pooling Layer #2
    conv2 = tf.layers.conv2d(
        inputs=pool1,
        filters=64,
        kernel_size=[5, 5],
        padding="same",
        activation=tf.nn.relu)

    print(template.format(conv2, conv2.shape))
    

    pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[32, 32], strides=32, padding="same")
    print(template.format(pool2, pool2.shape))


    # Dense Layer
    pool2_flat = tf.reshape(pool2, [-1, config.HIDDEN_UNITS])
    print(template.format(pool2_flat, pool2_flat.shape))

    d = tf.layers.dense(inputs=pool2_flat, units=config.HIDDEN_UNITS, activation=tf.nn.relu)
    print(template.format(d, d.shape))

    dense1 = tf.layers.dense(inputs=d, units=config.HIDDEN_UNITS, activation=tf.nn.relu)
    print(template.format(dense1, dense1.shape))


    # Logits Layer
    logits = tf.layers.dense(inputs=dense1, units=config.NUM_RAGAS, activation=None)
    print(template.format(logits, logits.shape))


    return tf.reshape(logits, [config.NUM_RAGAS, -1])
    # return logits

def dummy_model(x):
    w = tf.Variable(dtype=tf.float32, shape=(config.LENGTH, config.NUM_RAGAS), initial_value=tf.zeros(config.LENGTH, config.NUM_RAGAS))
    return tf.matmul(x, w)

def repeated_forward_prop(x):
    
    ans = tf.zeros(config.NUM_RAGAS)
    print("DEBUG: x shape: {}".format(x.shape))
    for i in range(2000):
        temp = x[:, i]
        temp = tf.reshape(temp, [-1, 129])
        ans += forward_prop(temp)

    # normalise the values of ans
    return tf.nn.softmax(ans)

def forward_prop(x):
    """ Vanilla Fully Connected Neural Network """
    # model = tf.layers.Dense(units=config.UNITS, activation=tf.nn.relu)
    model = tf.layers.Dense(units=config.UNITS, activation=tf.nn.relu)
    model = tf.layers.Dense(units=config.NUM_RAGAS, activation=None)
    return model(x)

def train_model():
    x = tf.placeholder(tf.float32, shape=(1, config.LENGTH))
    y_true = tf.placeholder(tf.int32)
    y_true = image_preprocess.conv_to_one_hot(y_true)

    y_pred = dummy_model(x)
    # y_pred = forward_prop(x)
    print('DEBUG: y_true shape: {}'.format(y_true))
    print('DEBUG: y_pred shape: {}'.format(y_pred))

    accuracy = tf.metrics.accuracy(labels=y_true, predictions=y_pred)
    loss = tf.losses.softmax_cross_entropy(onehot_labels=y_true, logits=y_pred)
    num_loss = tf.math.reduce_sum(loss)

    optimizer = tf.train.GradientDescentOptimizer(0.01)

    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)

    # tensorboard setup
    tf.summary.scalar('cross entropy loss', loss)
    tf.summary.scalar('Accuracy', accuracy)

    merged_summary_op = tf.summary.merge_all()

    template = ('Epoch: {}\tLoss: {}\tAccuracy: {}\n')

    summary_writer = tf.summary.FileWriter('logs', graph=tf.get_default_graph())

    init_op = tf.compat.v1.global_variables_initializer()
    # tf.Session().run(init_op)

    with tf.Session() as sess:
        sess.run(init_op)
        for i in range(10):
            a, b = get_data()

            for j in range(len(a)):
                # remove all examples with less than 2000 length
                # if a[key].shape[1] < config.BREADTH:
                #     continue

                _, quant_loss = sess.run((train, num_loss), feed_dict={x: a[j][:, 0], y_true: b[j]})
                val_accuracy = 0
                print(template.format(i, quant_loss, val_accuracy))

        # summary_writer.add_summary(summary, i)

    # testing code ??

    # Tensorboard functionality
    # writer = tf.summary.FileWriter('.')
    # writer.add_graph(tf.get_default_graph())
    # writer.flush()







# x, y_true = get_data()
if __name__ == "__main__":
    train_model()