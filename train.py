import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
import matplotlib.pyplot as plt
import os
import scipy.io
import cv2
import random

import src.deep_ensembels as de
import src.download_datasets as dd
from src import utils

#PULL INTO CONFIG FILE
# Parameters of data
img_size = 28
img_flat_size = img_size * img_size

num_label = 10 # 0 ~ 9

# Parameters of training
#Learning_rate = 0.0005
#epsilon = 1e-8

num_iter = 5000
batch_size = 256

validation_ratio = 0.1
gpu_fraction = 0.5

# Ensemble networks
networks = ['network1', 'network2', 'network3', 'network4', 'network5']

# Convolution [kernel size, kernel size, # input, # output]
#first_conv  = [3,3, 1,32]
#second_conv = [3,3,32,64]

# Dense [input size, output size]
#first_dense  = [7*7*64, num_label]


# Import MNIST dataset
(train_img, train_l), (testval_img, testval_l) = dd.mnist()

# Dataset for train, test, validation
test_len = testval_l.shape[0]
validation_len = int(test_len * validation_ratio)

train_x = train_img
test_x = testval_img[validation_len : test_len, :]
validation_x = testval_img[ : validation_len, :]

train_y_index = train_l
test_y_index = testval_l[validation_len : test_len]
validation_y_index = testval_l[ : validation_len]

train_y = np.zeros([train_y_index.shape[0], num_label])
test_y = np.zeros([test_y_index.shape[0], num_label])
validation_y = np.zeros([validation_y_index.shape[0], num_label])

for i in range(train_y.shape[0]):
    train_y[i, train_y_index[i]] = 1

for i in range(test_y.shape[0]):
    test_y[i, test_y_index[i]] = 1

for i in range(validation_y.shape[0]):
    validation_y[i, validation_y_index[i]] = 1

print("\nTraining X shape: " + str(train_x.shape))
print("Testing X shape: " + str(test_x.shape))
print("Validation X shape: " + str(validation_x.shape))

print("\nTraining Y shape: " + str(train_y.shape))
print("Testing Y shape: " + str(test_y.shape))
print("Validation Y shape: " + str(validation_y.shape))

(not_mnist_img, not_mnist_l) = dd.fashion_mnist()

not_mnist_x = not_mnist_img
not_mnist_y = np.zeros([not_mnist_l.shape[0], num_label])

for i in range(not_mnist_y.shape[0]):
    not_mnist_y[i, not_mnist_l[i]] = 1

print("not_mnist X shape: " + str(not_mnist_x.shape))
print("not_mnist Y shape: " + str(not_mnist_y.shape))

#Plotting Datasets
utils.plot_ex_dataset(train_x, 'train_data_ex.png', num_sample=5, img_size=img_size)
utils.plot_ex_dataset(not_mnist_x, 'not_train_data_ex.png', num_sample=5, img_size=img_size)

#STARTING TRAINING
tf.reset_default_graph()

#Initialize Ensemble Networks
x_list = []
y_list = []
output_list = []
loss_list = []
train_list = []
train_var_list = []

# Train each ensemble network
for i in range(len(networks)):
    x_image, y_label, output, loss, train_opt, train_vars = de.get_network(networks[i])

    x_list.append(x_image)
    y_list.append(y_label)
    output_list.append(output)
    loss_list.append(loss)
    train_list.append(train_opt)
    train_var_list.append(train_vars)

#Create Session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = gpu_fraction

sess = tf.InteractiveSession(config=config)
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()   
saver.save(sess, 'deep_ensembles_models/ses1')

#TRAIN
# Initialize data for printing
loss_check     = np.zeros(len(networks))
acc_check      = np.zeros(len(networks))
acc_check_test = np.zeros(len(networks))
acc_check_test_final = 0

# Set parameters for printing and testing
num_print = 100
test_size = 10

train_data_num = train_x.shape[0]
test_data_num  = test_x.shape[0]

for iter in range(num_iter):
    output_temp = np.zeros([test_size, num_label])

    # Making batches(testing)
    batch_x_test, batch_y_test = utils.making_batch(test_data_num, test_size, test_x, test_y)

    for i in range(len(networks)):
        # Making batches(training)
        batch_x, batch_y = utils.making_batch(train_data_num, batch_size, train_x, train_y)

        # Training
        _, loss, prob = sess.run([train_list[i], loss_list[i], output_list[i]],
                                 feed_dict = {x_list[i]: batch_x, y_list[i]: batch_y})

        # Testing
        loss_test, prob_test = sess.run([loss_list[i], output_list[i]],
                                         feed_dict = {x_list[i]: batch_x_test, y_list[i]: batch_y_test})

        # Add test prediction for get final prediction
        output_temp += prob_test

        # Calculate Accuracy (Training)
        acc_training = utils.get_accuracy(prob, batch_y)

        # Calculate Accuracy (testing)
        acc_testing = utils.get_accuracy(prob_test, batch_y_test)

        # Get accuracy and loss for each network
        acc_check[i] += acc_training
        acc_check_test[i] += acc_testing
        loss_check[i] += loss

    # Get final test prediction
    prob_test_final = output_temp / len(networks)

    # Calculate Accuracy (Testing final)
    acc_testing_final = utils.get_accuracy(prob_test_final, batch_y_test)
    acc_check_test_final += acc_testing_final

    if iter % num_print == 0 and iter != 0:
        print(('-------------------------') + ' Iteration: ' + str(iter) + ' -------------------------')
        print('Average Loss(Brier score): ' + str(loss_check / num_print))
        print('Training Accuracy: ' + str(acc_check / num_print))
        print('Testing Accuracy: ' + str(acc_check_test / num_print))
        print('Final Testing Accuracy: ' + str(acc_check_test_final / num_print))
        print('\n')

        loss_check = np.zeros(len(networks))
        acc_check = np.zeros(len(networks))
        acc_check_test = np.zeros(len(networks))
        acc_check_test_final = 0

saver.save(sess, 'deep_ensembles_models/sess1')
