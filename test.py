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

#PARAMS
img_size = 28
img_flat_size = img_size * img_size
num_label = 10 # 0 ~ 9
networks = ['network1', 'network2', 'network3', 'network4', 'network5']
validation_ratio = 0.1
gpu_fraction = 0.5

(_,_),(mnist_img, mnist_l) = dd.mnist()
mnist_x = mnist_img
mnist_y = np.zeros([mnist_l.shape[0], num_label])

for i in range(mnist_y.shape[0]):
    mnist_y[i, mnist_l[i]] = 1

print("mnist X shape: " + str(mnist_x.shape))
print("mnist Y shape: " + str(mnist_y.shape))

(not_mnist_img, not_mnist_l) = dd.fashion_mnist()

not_mnist_x = not_mnist_img
not_mnist_y = np.zeros([not_mnist_l.shape[0], num_label])

for i in range(not_mnist_y.shape[0]):
    not_mnist_y[i, not_mnist_l[i]] = 1

print("not_mnist X shape: " + str(not_mnist_x.shape))
print("not_mnist Y shape: " + str(not_mnist_y.shape))

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
saver.restore(sess, 'deep_ensembles_models/sess1')

#WANT TO TURN INTO FUNCTIONS

#Testing with MNIST        
MNIST_sample_index = np.random.choice(mnist_x.shape[0], 10)

# Test with training plotting
f_Mnist, ax_Mnist = plt.subplots(1, 10, figsize=(12,24))

output_sample     = []
output_sample_one = []

for i, index in enumerate(MNIST_sample_index):
    img_MNIST = np.reshape(mnist_x[index], (img_size, img_size))
    
    ax_Mnist[i].imshow(img_MNIST, cmap='gray')
    ax_Mnist[i].axis('off')
    ax_Mnist[i].set_title(str(i+1) + 'th')
    
    output_test = np.zeros([1, num_label])
    
    for net_index in range(len(networks)):
        x_temp = np.reshape(mnist_x[index, :], (1, img_size, img_size, 1))
        y_temp = np.reshape(mnist_y[index, :], (1, num_label))
        
        loss_temp, prob_temp = sess.run([loss_list[net_index], output_list[net_index]], 
                                         feed_dict = {x_list[net_index]: x_temp, y_list[net_index]: y_temp})
        
        # Add test prediction for get final prediction
        output_test += prob_temp
        
    # Get final test prediction
    prob_temp_final = output_test / len(networks)
    prob_temp_one   = prob_temp
    
    output_sample.append(prob_temp_final)
    output_sample_one.append(prob_temp_one)

plt.savefig('images/trained_img.png')

print("====================== Ensemble Result ======================")
array_ensemble_MNIST = np.zeros([10])
for i in range(len(output_sample)):

    idx_sample = np.argmax(output_sample[i])
    max_prob = output_sample[i][0, idx_sample]
    array_ensemble_MNIST[i] = max_prob
    
    print(str(i+1) + 'th sample: label = ' + str(idx_sample) + ', Probability = ' + str(max_prob))

print("\n====================== SingleNet Result ======================")
array_single_MNIST = np.zeros([10])
for i in range(len(output_sample_one)):
    
    idx_sample = np.argmax(output_sample_one[i])
    max_prob = output_sample_one[i][0, idx_sample]
    array_single_MNIST[i] = max_prob
    
    print(str(i+1) + 'th sample: label = ' + str(idx_sample) + ', Probability = ' + str(max_prob))
    
plt.figure()
plt.plot(array_ensemble_MNIST, 'or')
plt.plot(array_single_MNIST, 'ob')
plt.xlim([-1, 10])
plt.ylim([np.min(array_single_MNIST) - 0.005, np.max(array_ensemble_MNIST) + 0.005])
plt.xlabel('Sample Index')
plt.ylabel('Probability')
plt.legend(['Ensemble', 'Single'], loc='best')
plt.savefig('images/trained_results.png')

#Test with not training
not_mnist_sample_index = np.random.choice(not_mnist_x.shape[0], 10)

# MNIST plotting
f_not_mnist, ax_not_mnist = plt.subplots(1, 10, figsize=(12,24))

output_sample     = []
output_sample_one = []

for i, index in enumerate(not_mnist_sample_index):
    img_not_mnist = np.reshape(not_mnist_x[index], (img_size, img_size))
    
    ax_not_mnist[i].imshow(img_not_mnist, cmap='gray')
    ax_not_mnist[i].axis('off')
    ax_not_mnist[i].set_title(str(i+1) + 'th')
    
    output_test = np.zeros([1, num_label])
    
    for net_index in range(len(networks)):
        x_temp = np.reshape(not_mnist_x[index, :], (1, img_size, img_size, 1))
        y_temp = np.reshape(not_mnist_y[index, :], (1, num_label))
        
        loss_temp, prob_temp = sess.run([loss_list[net_index], output_list[net_index]], 
                                         feed_dict = {x_list[net_index]: x_temp, y_list[net_index]: y_temp})
        
        # Add test prediction for get final prediction
        output_test += prob_temp
        
    # Get final test prediction
    prob_temp_final = output_test / len(networks)
    prob_temp_one   = prob_temp

    output_sample.append(prob_temp_final)
    output_sample_one.append(prob_temp_one)

plt.savefig('images/untrained_img.png')

print("====================== Ensemble Result ======================")
array_ensemble_not_mnist = np.zeros([10])
for i in range(len(output_sample)):
    
    idx_sample = np.argmax(output_sample[i])
    max_prob = output_sample[i][0, idx_sample]
    array_ensemble_not_mnist[i] = max_prob
    
    print(str(i+1) + 'th sample: label = ' + str(idx_sample) + ', Probability = ' + str(max_prob))

print("\n====================== SingleNet Result ======================")
array_single_not_mnist = np.zeros([10])
for i in range(len(output_sample_one)):
    
    idx_sample = np.argmax(output_sample_one[i])
    max_prob = output_sample_one[i][0, idx_sample]
    array_single_not_mnist[i] = max_prob
    
    print(str(i+1) + 'th sample: label = ' + str(idx_sample) + ', Probability = ' + str(max_prob))
    
