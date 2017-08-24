'''
A logistic regression learning algorithm example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report




factors=pd.read_csv("data/train_factors.csv")
test_factorscsv=pd.read_csv("data/test_factors.csv")
labels=pd.read_csv('data/train_labels.csv')
predictors = ["SEX", "EDUCATION", "MARRIAGE","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5",
"BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
"PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","AGE",
 "CRED_LIMIT"]


def transdata(csvdata):
    data_old=csvdata.values;
    data_new=[]
    for i,data in enumerate(data_old):
        data_new.append(list(data[0:]))

    return data_new;

trans_factors=factors[predictors].values
enc = preprocessing.OneHotEncoder(categorical_features=np.array([0,1,2]))
enc.fit(trans_factors)
nor_factors=enc.transform(trans_factors).toarray();
print(trans_factors[0])
print(nor_factors[0])

nor_factors2=enc.transform(test_factorscsv[predictors].values).toarray();

train_factors=preprocessing.scale(nor_factors);
train_labels=transdata(labels);
test_factors=nor_factors[16001:];
test_labels=transdata(labels)[16001:];
test_factors2=nor_factors2;



print(test_factors)
print(test_labels)


#mnist = input_data.read_data_sets("data/", one_hot=True)

# Parameters
learning_rate = 0.0001
training_epochs = 25
batch_size =5000
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 28]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 2]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([28, 2]))
b = tf.Variable(tf.zeros([2]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
# saver=tf.train.Saver(tf.global_variables())

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(train_factors)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            min_max_scaler = preprocessing.MinMaxScaler()
            batch_xs = np.array(train_factors[i:(i+1)*batch_size])
            batch_ys =  np.array(train_labels[i:(i+1)*batch_size])
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_xs,
                                                          y: batch_ys})
            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch+1) % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
        # if epoch%5==0:
        #     print(i,avg_cost)
        #     print("保存模型：",saver.save(sess,'stock.model'))  

    print("Optimization Finished!")

    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    pred_result=tf.argmax(pred, 1).eval({x:preprocessing.scale(test_factors)})
    real_result=tf.argmax(y,1).eval({y:test_labels})
    print(classification_report(real_result,pred_result))
    test_result=pred.eval({x:preprocessing.scale(test_factors2)})
    result = pd.DataFrame(test_result)
    print(len(test_result))
    result.to_csv('F:\GitHub\ML\data\\result_mt3.csv')


    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: test_factors, y: test_labels}))

    