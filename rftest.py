# -*- coding:utf8 -*-

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("E:\machineLearnning\demo\data\\train.csv")
test = pd.read_csv("E:\machineLearnning\demo\data\\test.csv")


# def harmonize_data(titanic):
#     # 填充空数据 和 把string数据转成integer表示

#     titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())

#     titanic.loc[titanic["Sex"] == "male", "Sex"] = 0
#     titanic.loc[titanic["Sex"] == "female", "Sex"] = 1

#     titanic["Embarked"] = titanic["Embarked"].fillna("S")

#     titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
#     titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
#     titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2

#     titanic["Fare"] = titanic["Fare"].fillna(titanic["Fare"].median())

#     return titanic

train_data = train

predictors = ["PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
"BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
"PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6",
"SEX", "AGE", "CRED_LIMIT", "EDUCATION", "MARRIAGE"]
results = []
sample_leaf_options = list(range(21, 500, 5))
n_estimators_options = list(range(1, 200, 5))
groud_truth = train_data['Default'][16001:]

# for leaf_size in sample_leaf_options:
#     for n_estimators_size in n_estimators_options:
#         alg = RandomForestClassifier(min_samples_leaf=leaf_size, n_estimators=n_estimators_size, random_state=40)
#         alg.fit(train_data[predictors][:16000], train_data['Default'][:16000])
#         predict = alg.predict(train_data[predictors][16001:])
#         # 用一个三元组，分别记录当前的 min_samples_leaf，n_estimators， 和在测试数据集上的精度
#         results.append((leaf_size, n_estimators_size, (groud_truth == predict).mean()))
#         # 真实结果和预测结果进行比较，计算准确率
#         print((groud_truth == predict).mean())

# 打印精度最大的那一个三元组
# print(max(results, key=lambda x: x[2]))


alg = RandomForestClassifier(min_samples_leaf=21, n_estimators=116, random_state=40)
alg.fit(train_data[predictors][:16000], train_data['Default'][:16000])
predict = alg.predict(train_data[predictors][16001:])
print((groud_truth == predict).mean())
from sklearn.metrics import classification_report
print(classification_report(groud_truth,predict))

# alg = RandomForestClassifier(min_samples_leaf=21, n_estimators=116, random_state=40)
# alg.fit(train_data[predictors], train_data['Default'])
# predict = alg.predict(test[predictors])
# result = pd.DataFrame(predict)

# result.to_csv('result_mt1.csv')
