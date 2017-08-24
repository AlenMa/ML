import tensorflow as tf
import numpy as np
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import classification_report
#
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


# train_factors=preprocessing.scale(transdata(factors));
# train_labels=transdata(labels);
# test_factors=transdata(factors)[16001:];
# test_labels=transdata(labels)[16001:];
# test_factors2=transdata(test_factorscsv);

# 添加层
def add_layer(inputs, in_size, out_size, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    # 在 Wx_plus_b 上drop掉一定比例
    # keep_prob 保持多少不被drop，在迭代时在 sess.run 中 feed
    Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

# 1.训练的数据
# Make up some real data 


# 2.定义节点准备接收数据
# define placeholder for inputs to network  
xs = tf.placeholder(tf.float32, [None, 28])
ys = tf.placeholder(tf.float32, [None, 2])
keep_prob = tf.placeholder("float")

# 3.定义神经层：隐藏层和预测层
# add hidden layer 输入值是 xs，在隐藏层有 10 个神经元   
l1 = add_layer(xs, 28, 7, activation_function=tf.nn.softmax)
# add output layer 输入值是隐藏层 l1，在预测层输出 1 个结果
prediction = add_layer(l1,7, 2, activation_function=tf.nn.softmax)

# 4.定义 loss 表达式
# the error between prediciton and real data    
# the error between prediction and real data
# loss 函数用 cross entropy
loss = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))       # loss
train_step = tf.train.GradientDescentOptimizer(0.00001).minimize(loss)




# important step 对所有变量进行初始化
init = tf.initialize_all_variables()
sess = tf.Session()
# 上面定义的都没有运算，直到 sess.run 才会开始运算
sess.run(init)

# 迭代 1000 次学习，sess.run optimizer
for i in range(1000):
    # training train_step 和 loss 都是由 placeholder 定义的运算，所以这里要用 feed 传入参数
    batch_xs = np.array(train_factors[i:(i+1)*10000])
    batch_ys =  np.array(train_labels[i:(i+1)*10000])
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:0.4})
    
    # to see the step improvement
    print(sess.run(loss, feed_dict={xs: batch_xs, ys: batch_ys,keep_prob:0.4}))

correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(ys, 1))
pred_result=tf.argmax(prediction, 1).eval({xs:preprocessing.scale(test_factors),keep_prob:0.4},session=sess)
real_result=tf.argmax(ys,1).eval({ys:test_labels},session=sess)
print(classification_report(real_result,pred_result))
test_result=tf.argmax(prediction,1).eval({xs:preprocessing.scale(test_factors2),keep_prob:0.4},session=sess)
result = pd.DataFrame(test_result)
print(len(test_result))
result.to_csv('F:\GitHub\ML\data\\result_mt2.csv')


    # Calculate accuracy
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("Accuracy:", accuracy.eval({xs: preprocessing.scale(test_factors), ys: test_labels,keep_prob:0.4},session=sess))

