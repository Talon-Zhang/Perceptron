# perceptron.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/27/2018

"""
This is the main entry point for MP6. You should only modify code
within this file -- the unrevised staff files will be used for all other
files and classes when code is run, so be careful to not modify anything else.
"""
import numpy as np
# import time

def classify(train_set, train_labels, dev_set, learning_rate,max_iter):
    """
    train_set - A Numpy array of 32x32x3 images of shape [7500, 3072].
                This can be thought of as a list of 7500 vectors that are each
                3072 dimensional.  We have 3072 dimensions because there are
                each image is 32x32 and we have 3 color channels.
                So 32*32*3 = 3072
    train_labels - List of labels corresponding with images in train_set
    example: Suppose I had two images [X1,X2] where X1 and X2 are 3072 dimensional vectors
             and X1 is a picture of a dog and X2 is a picture of an airplane.
             Then train_labels := [1,0] because X1 contains a picture of an animal
             and X2 contains no animals in the picture.

    dev_set - A Numpy array of 32x32x3 images of shape [2500, 3072].
              It is the same format as train_set
    """
    # TODO: Write your code here
    # return predicted labels of development set

    # initialize the weights. The absolute value ranges are between [0,1).
    # The sign of the weights are determined randomly as well
    weights=np.zeros(3073)
    for i in range(3073):
    	if np.random.randint(0,2)==0:
    		weights[i]=-1*np.random.rand()
    	else:
    		weights[i]=np.random.rand()

    # add an extra 1 to the end of the input. So the train_set now is in shape (7500,3073)
    addition = np.full(len(train_set), 1)
    new_input = np.hstack((train_set, addition[:,None]))

    # looping through the train_set
    for k in range(0, max_iter):
    	#start_time = time.time()
    	for i in range(0,len(train_set)):
    		image = new_input[i]
    		#out is -1 if the sign is negative, 1 if the sign is positive
    		out = np.sign(np.dot(weights,image))
    		if train_labels[i] == 0:
    			supposed_out = -1
    		else:
    			supposed_out = 1

    		if out != supposed_out:
    			weights = weights+learning_rate*supposed_out*image

    	#end_time = time.time()
    	#print(str(end_time-start_time)+' sec')


    # now have the updated weights
    predicts = []
    # again add an extra 1 to the end
    addition2 = np.full(len(dev_set), 1)
    new_dev = np.hstack((dev_set, addition2[:,None]))
    for j in range(0, len(new_dev)):
    	output = np.dot(new_dev[j], weights)
    	if np.sign(output) == -1:
    		predicts.append(0)
    	else:
    		predicts.append(1)

    return predicts


def classifyEC(train_set, train_labels, dev_set, learning_rate, max_iter):
    # Write your code here if you would like to attempt the extra credit
    # start_time = time.time()

    predicts = []
    for dev in dev_set:
        distance_list = []
        for train in train_set:
            distance = np.sum(np.absolute(np.subtract(train, dev)))
            distance_list.append(distance)

        distance_list = np.array(distance_list)
        nearest = np.argpartition(distance_list, 5)[:5]
        true_num = 0
        false_num = 0
        for j in nearest:
            if train_labels[j] == 1:
                true_num += 1
            else:
                false_num += 1
        if true_num > false_num:
            predicts.append(1)
        else:
            predicts.append(0)

    # end_time = time.time()
    # print(str(end_time-start_time)+' sec')

    return predicts
