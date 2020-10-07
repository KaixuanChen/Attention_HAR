# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function

import tensorflow as tf
from C_A_L_model import RecurrentAttentionModel
import logging
import numpy as np
from Data_Processing import *
import os
from copy import deepcopy
# import scipy.io as sc
import time

# logging.getLogger().setLevel(logging.INFO)
isTrain = True
test_subject = 2
print('haha')
# checkpoint_dir = './sub' + str(test_subject) + '_12classes_0000/'
# checkpoint_dir = './sub' + str(test_subject) + '_try/'
checkpoint_dir = './sub' + str(test_subject) + '_0002/'
print(checkpoint_dir)

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

learning_rate = 1e-3
learning_rate_decay_factor = 0.97
min_learning_rate = 1e-5
max_gradient_norm = 5.0
num_steps = 100000

CNN_patch_width = 3
CNN_patch_height = 3
CNN_patch_number = 32

patch_window_width = 16 # size of glimpse window size
patch_window_height = 64
g_size = 128 # Size of theta_g^0
l_size = 128 # Size of theta_g^1
glimpse_output_size = 220 # Output size of Glimpse Network
cell_size = 100 # Size of LSTM cell
num_glimpses = 30 # Number of glimpses
variance = 0.22 # Gaussian variance for Location Network
M = 20 # Monte Carlo sampling, see Eq(2)

n_time_window = 5
last_lstm_size = 1000

feature_num = 702
training_data_size = 80000
validation_data_size = 20000
test_data_size = 10000
batch_size = 1000
training_batch_num = training_data_size // batch_size
test_batch_num = test_data_size // batch_size
class_num = 6

if isTrain:

    file_name = 'PAMAP2_Protocol_6_classes_balance'
    file_data = sc.loadmat(file_name+'.mat')
    file_data = file_data[file_name]
    start = np.where(file_data[:, -1] == test_subject)[0][0] # file_data[:,-1]: subjects
    end = np.where(file_data[:, -1] == test_subject)[0][-1] + 1
    training_data = np.append(file_data[: start], file_data[end:], axis=0) # leave-one-subject-out cross validation 
    test_data = file_data[start: end]

    training_data = extract_time_sequences(training_data, n_time_window,
                                           (training_data_size + validation_data_size) // n_time_window)
    test_data = extract_time_sequences(test_data, n_time_window, test_data_size // n_time_window)

    # np.random.shuffle(training_data)
    # np.random.shuffle(test_data)

    tmp = training_data
    training_data = tmp[0: training_data_size]
    # validation_data = tmp[training_data_size: training_data_size + validation_data_size]
    test_data = test_data[0: test_data_size]

    training_data = PAMAP_Image(training_data) # preproccessing
    # validation_data = PAMAP_Image(validation_data)
    test_data = PAMAP_Image(test_data)

    sc.savemat(checkpoint_dir+'training_data.mat', {'training_data':training_data})
    sc.savemat(checkpoint_dir+'test_data.mat', {'test_data':test_data})
else:
    training_data = sc.loadmat(checkpoint_dir + 'training_data.mat')
    training_data = training_data['training_data']
    test_data = sc.loadmat(checkpoint_dir + 'test_data.mat')
    test_data = test_data['test_data']


training_feature = training_data[:, 0: feature_num]
# validation_feature = validation_data[:, 0: feature_num]
test_feature = test_data[:, 0: feature_num]
training_label = training_data[:, -2] # data[:,-2]: label
# validation_label = validation_data[:, -2]
test_label = test_data[:, -2]

training_label = training_label.astype(int)
# validation_label = validation_label.astype(int)
test_label = test_label.astype(int)


training_feature_batch = []  # list
for i in range(training_batch_num):
    one_batch_data = training_feature[batch_size * i: batch_size * (i + 1)]
    one_batch_data = np.tile(one_batch_data, [M, 1]) # (M*batch_num, 702)
    training_feature_batch.append(one_batch_data)

training_label_batch = []  # list
for i in range(training_batch_num):
    one_batch_data = training_label[batch_size * i: batch_size * (i + 1)]
    one_batch_data = np.tile(one_batch_data, [M]) # (M*batch_num, 1)
    training_label_batch.append(one_batch_data)

test_feature_batch = [] # list
for i in range(test_batch_num):
    one_batch_data = test_feature[batch_size * i : batch_size * (i + 1)]
    test_feature_batch.append(one_batch_data)

test_label_batch = [] # list
for i in range(test_batch_num):
    one_batch_data = test_label[batch_size * i : batch_size * (i + 1)]
    test_label_batch.append(one_batch_data)

ram = RecurrentAttentionModel(img_size_width=9,
                              img_size_height = 78,
                              CNN_patch_width = CNN_patch_width,
                              CNN_patch_height = CNN_patch_height,
                              CNN_patch_number = CNN_patch_number,
                              patch_window_width = patch_window_width,
                              patch_window_height=patch_window_height,
                              g_size=g_size,
                              l_size=l_size,
                              glimpse_output_size=glimpse_output_size,
                              loc_dim=2,   # (x,y)
                              variance=variance,
                              cell_size=cell_size,
                              num_glimpses=num_glimpses,
                              num_classes=class_num,
                              learning_rate=learning_rate,
                              learning_rate_decay_factor=learning_rate_decay_factor,
                              min_learning_rate=min_learning_rate,
                              training_batch_num=training_batch_num,
                              max_gradient_norm=max_gradient_norm,
                              last_lstm_size=last_lstm_size,
                              n_time_window=n_time_window,
                              is_training=True)

# images_training, labels_training = training_feature_batch[0], training_label_batch[0]
# labels_bak_training = labels_training

# images_validation, labels_validation = validation_feature, validation_label
# labels_bak_validation = labels_validation

images_test, labels_test = test_feature_batch, test_label_batch
labels_bak_test = deepcopy(labels_test)

# Duplicate M times
# images_training = np.tile(images_training, [M, 1])
# labels_training = np.tile(labels_training, [M])

# images_validation = np.tile(images_validation, [M, 1])
# labels_validation = np.tile(labels_validation, [M])

images_test = np.tile(images_test, [M, 1])
labels_test = np.tile(labels_test, [M])

epoch = 0
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
saver = tf.train.Saver()

with tf.Session(config=config) as sess:
    if not isTrain:
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            # softmax_training = sess.run(ram.softmax,
            #                             feed_dict={
            #                                 ram.img_ph: images_training,
            #                                 ram.lbl_ph: labels_training
            #                             })
            # softmax_validation = sess.run(ram.softmax,
            #                               feed_dict={
            #                                   ram.img_ph: images_validation,
            #                                   ram.lbl_ph: labels_validation
            #                               })
            # softmax_validation = np.reshape(softmax_validation, [M, -1, 6])
            # softmax_validation = np.mean(softmax_validation, 0)
            # prediction_validation = np.argmax(softmax_validation, 1).flatten()
            # acc_validation = np.sum(prediction_validation ==
            #                         labels_bak_validation) / validation_data_size

            acc_test = 0
            # time_1 = time.clock()


            for j in range(test_batch_num):
                softmax_test = sess.run(ram.softmax,
                                        feed_dict={
                                            ram.img_ph: images_test[j],
                                            ram.lbl_ph: labels_test[j]
                                        })
                softmax_test = np.reshape(softmax_test, [M, -1, class_num])
                softmax_test = np.mean(softmax_test, 0)
                prediction_test = np.argmax(softmax_test, 1).flatten()
                acc_test += np.sum(prediction_test ==
                                  labels_bak_test[j]) / batch_size
            acc_test /= test_batch_num

            # glimpse_list = sess.run(ram.glimpse_list,
            #                             feed_dict={
            #                                 ram.img_ph: images_test[0],
            #                                 ram.lbl_ph: labels_test[0]
            #                             })
            # print('glimpse_list', glimpse_list)
            # sc.savemat('glimpse_list.mat', {'glimpse_list': glimpse_list})
            # sc.savemat('test_image.mat', {'test_image': images_test[0]})

            # print('locs:',sess.run(ram.locs,
            #                             feed_dict={
            #                                 ram.img_ph: images_test[0],
            #                                 ram.lbl_ph: labels_test[0]
            #                             }))
            # print('label',labels_test[0])

            confusion_matrix = [[0] * class_num for _ in range(class_num)]
            for i in range(test_data_size):
                confusion_matrix[labels_bak_test[j][i]][prediction_test[i]] += 1
            for i in range(class_num):
                confusion_matrix[i] = [100 * j / sum(confusion_matrix[i]) for j in confusion_matrix[i]]
            print('confusion_matrix:')
            print(confusion_matrix)
            sc.savemat('confusion_matrix.mat', {'confusion_matrix': confusion_matrix})

            # time_2 = time.clock()
            # print('time: ', time_2 - time_1)

            # softmax_training = np.reshape(softmax_training, [M, -1, 6])
            # softmax_training = np.mean(softmax_training, 0)
            # prediction_training = np.argmax(softmax_training, 1).flatten()
            # acc_training = np.sum(prediction_training ==
            #                       labels_bak_training) / batch_size




            print(epoch,
                  # 'training = ', '%.5f' % (acc_training),
                  # 'validation = ', '%.5f' % (acc_validation),
                  'test = ', '%.5f' % (acc_test), '\n')


    else:
        sess.run(tf.global_variables_initializer())
        result_list = [[],[],[]]
        mean_10_list = []
        while epoch <= 10000000:
            for i in range(training_batch_num):
                images, labels = training_feature_batch[i], training_label_batch[i]
                # images = np.tile(images, [M, 1]) # (320, 784)
                # labels = np.tile(labels, [M]) # (320,)

                output_feed = [ram.train_op, ram.loss,
                            ram.cross_entropy, ram.reward,
                             ram.advantage, ram.baselines_mse,
                             ram.learning_rate]
                _, loss, cross_entropy, reward, advantage, baselines_mse, learning_rate = sess.run(output_feed,
                                                                                 feed_dict={
                                                                                      ram.img_ph: images,
                                                                                      ram.lbl_ph: labels
                                                                                    })

                # Evaluation
                if epoch >= 1 and i % 100 == 0:
                    # softmax_validation = sess.run(ram.softmax,
                    #                               feed_dict={
                    #                                   ram.img_ph: images_validation,
                    #                                   ram.lbl_ph: labels_validation
                    #                               })
                    # softmax_validation = np.reshape(softmax_validation, [M, -1, 6])
                    # softmax_validation = np.mean(softmax_validation, 0)
                    # prediction_validation = np.argmax(softmax_validation, 1).flatten()
                    # acc_validation = np.sum(prediction_validation ==
                    #                         labels_bak_validation) / validation_data_size

                    acc_test = 0
                    for j in range(test_batch_num):
                        softmax_test = sess.run(ram.softmax,
                                                feed_dict={
                                                    ram.img_ph: images_test[j],
                                                    ram.lbl_ph: labels_test[j]
                                                })
                        softmax_test = np.reshape(softmax_test, [M, -1, class_num])
                        softmax_test = np.mean(softmax_test, 0)
                        prediction_test = np.argmax(softmax_test, 1).flatten()
                        acc_test += np.sum(prediction_test ==
                                           labels_bak_test[j]) / batch_size
                    acc_test /= test_batch_num
                    



                    # print(i)
            if epoch >= 1:
                print(training_data_size,checkpoint_dir)
                print(epoch, 'loss', loss,
                      # 'training = ', '%.5f' % (acc_training),
                      # 'validation = ', '%.5f' % (acc_validation),
                      'test = ', '%.5f' % (acc_test), '\n')

                
            epoch += 1
