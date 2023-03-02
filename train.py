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

epoch = 5000
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

class Train():
    def __init__(self):
        #CONFIG STUFF
        self.epoch = 5000
        self.batch_size = 256
        self.validation_ratio = 0.1
        self.gpu_fraction = 0.5
        self.networks = ['network1', 'network2', 'network3', 'network4', 'network5']
        
        self.get data()
    
        #Plotting Datasets
        utils.plot_ex_dataset(self.train_x, 'train_data_ex.png', num_sample=5, img_size=img_size)

        #STARTING TRAINING
        tf.reset_default_graph()

        #Initialize Ensemble Networks
        self.x_list = []
        self.y_list = []
        self.output_list = []
        self.loss_list = []
        self.train_list = []
        self.train_var_list = []

        # Train each ensemble network
        for i in range(len(self.networks)):
            x_image, y_label, output, loss, train_opt, train_vars = de.get_network(self.networks[i])

            self.x_list.append(x_image)
            self.y_list.append(y_label)
            self.output_list.append(output)
            self.loss_list.append(loss)
            self.train_list.append(train_opt)
            self.train_var_list.append(train_vars)

        #Create Session
        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = self.gpu_fraction

        self.sess = tf.InteractiveSession(config=config)
        self.sess.run(tf.global_variables_initializer())
        self.saver = tf.train.Saver()   

        #TRAIN
        # Initialize data for printing
        loss_check     = np.zeros(len(self.networks))
        acc_check      = np.zeros(len(self.networks))
        acc_check_test = np.zeros(len(self.networks))
        acc_check_test_final = 0

        # Set parameters for printing and testing
        num_print = 100
        test_size = 10

        train_data_num = train_x.shape[0]
        test_data_num  = test_x.shape[0]

        for iter in range(self.epoch):
            output_temp = np.zeros([test_size, num_label])

            # Making batches(testing)
            batch_x_test, batch_y_test = utils.making_batch(test_data_num, test_size, test_x, test_y)

            for i in range(len(self.networks)):
                # Making batches(training)
                batch_x, batch_y = utils.making_batch(train_data_num, self.batch_size, train_x, train_y)

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
            prob_test_final = output_temp / len(self.networks)

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

                loss_check = np.zeros(len(self.networks))
                acc_check = np.zeros(len(self.networks))
                acc_check_test = np.zeros(len(self.networks))
                acc_check_test_final = 0

        self.saver.save(sess, 'deep_ensembles_models/sess1')
        
    def get_data(self):
        '''
        Function to bring in the data for the training session
        Currently hard coded for MNIST
        '''
        (self.train_x, self.train_y), (self.validation_x, self.validation_y),(test_x, self.test_y) = dd.mnist(self.validation_ratio)
        print("\nPrinting MNIST dataset shape") #Replace MNIST with a string variable
        print("\nTraining X shape: " + str(train_x.shape))
        print("Testing X shape: " + str(test_x.shape))
        print("Validation X shape: " + str(validation_x.shape))

        print("\nTraining Y shape: " + str(train_y.shape))
        print("Testing Y shape: " + str(test_y.shape))
        print("Validation Y shape: " + str(validation_y.shape) + '\n')
        
if __name__=='__main__':
    Train()
