import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

num_label = 10 # 0 ~ 9

# Ensemble networks
#networks = ['network1', 'network2', 'network3', 'network4', 'network5']

# Convolution [kernel size, kernel size, # input, # output]
first_conv  = [3,3, 1,32]
second_conv = [3,3,32,64]

# Dense [input size, output size]
first_dense  = [7*7*64, num_label]

# Parameters of training
Learning_rate = 0.0005
epsilon = 1e-8

#num_iter = 5000
#batch_size = 256

#validation_ratio = 0.1
#gpu_fraction = 0.5

# function for conv2d
def conv2d(x,w, stride):
	return tf.nn.conv2d(x,w,strides=[1, stride, stride, 1], padding='SAME')

# function for max pool
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# Get Variables
def conv_variable(name, shape):
    #return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer_conv2d())
    return tf.get_variable(name, shape = shape, initializer = tf.initializers.glorot_normal(seed=None))

def weight_variable(name, shape):
    #return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())
    return tf.get_variable(name, shape = shape, initializer = tf.initializers.glorot_normal(seed=None))

def bias_variable(name, shape):
    #return tf.get_variable(name, shape = shape, initializer = tf.contrib.layers.xavier_initializer())
    return tf.get_variable(name, shape = shape, initializer = tf.initializers.glorot_normal(seed=None))

# Get networks
def get_network(network_name, img_size=28):
    x_image = tf.placeholder(tf.float32, shape = [None, img_size, img_size, 1])

    with tf.variable_scope(network_name):
        # Convolution variables
        w_conv1 = conv_variable(network_name + '_w_conv1', first_conv)
        b_conv1 = bias_variable(network_name + '_b_conv1', [first_conv[3]])

        w_conv2 = conv_variable(network_name + '_w_conv2', second_conv)
        b_conv2 = bias_variable(network_name + '_b_conv2', [second_conv[3]])

        # Densely connect layer variables
        w_fc1 = weight_variable(network_name + '_w_fc1', first_dense)
        b_fc1 = bias_variable(network_name + '_b_fc1', [first_dense[1]])


    # Network
    h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1, 1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2, 1) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, [-1, first_dense[0]])

    logits = tf.matmul(h_pool2_flat, w_fc1) + b_fc1
    output = tf.nn.softmax(logits)

    y_label = tf.placeholder(tf.float32, shape = [None, num_label])

    # Brier Score
    loss = tf.reduce_mean(tf.div(tf.reduce_sum(tf.square(tf.subtract(output, y_label)), axis = 1), num_label), axis = 0)

    # Get trainable variables
    train_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, network_name)

    train_opt = tf.train.AdamOptimizer(Learning_rate).minimize(loss, var_list = train_vars)

    return x_image, y_label, output, loss, train_opt, train_vars
