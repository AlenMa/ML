import pandas as pd
import numpy as np

from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier 
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report


def deal_data(tranfile,predfile,predictors,trainnum):
    filedata1=pd.read_csv(tranfile)
    filedata2=pd.read_csv(predfile)
    traindata=filedata1[predictors].values
    trainlabel=filedata1['Default'].values
    preddata=filedata2[predictors].values

    return traindata,trainlabel,preddata

def accuracy(test_labels, pred_lables):  
    correct = np.sum(test_labels == pred_lables)
    n = len(test_labels)
    print(classification_report(test_labels,pred_lables))
    return float(correct) / n


def testLR(trandata,trainlabel,preddata):
    kf = KFold(len(trandata), n_folds=3, shuffle=True)  
    clf = LogisticRegression()
    result_set = [(clf.fit(trandata[train], trainlabel[train]).predict(trandata[test]), test) for train, test in kf]  
    score = [accuracy(trainlabel[result[1]], result[0]) for result in result_set]  
    print(score)

def testNaiveBayes(features, labels):
    kf = KFold(len(features), n_folds=3, shuffle=True)  
    clf = GaussianNB()
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]  
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]  
    print(score)

def testKNN(features, labels):
    kf = KFold(len(features), n_folds=3, shuffle=True)  
    clf = KNeighborsClassifier(n_neighbors=5) 
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]  
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]  
    print(score)

def testSVM(features, labels):
    kf = KFold(len(features), n_folds=3, shuffle=True)  
    clf = svm.SVC()
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]  
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]  
    print(score)

def testDecisionTree(features, labels):
    kf = KFold(len(features), n_folds=3, shuffle=True)  
    clf = DecisionTreeClassifier()
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]  
    score = [accuracy(labels[result[1]], result[0]) for result in result_set]  
    print(score)

def testRandomForest(features, labels,preddata):
    kf = KFold(len(features), n_folds=3, shuffle=True)  
    clf = RandomForestClassifier(min_samples_leaf=30, n_estimators=160, random_state=50,oob_score=True)
    result_set = [(clf.fit(features[train], labels[train]).predict(features[test]), test) for train, test in kf]
    i=0
    for train,test in kf:
        clf.fit(features[train],labels[train])
        predtest=clf.predict(features[test])
        print(classification_report(labels[test],predtest))
        pred=clf.predict(preddata)
        result=pd.DataFrame(pred)
        result.to_csv('result_mt'+str(i)+'.csv')
        i=i+1


    

if __name__ == '__main__':
    predictors = ["SEX", "EDUCATION", "MARRIAGE","PAY_1","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6","AGE",
    "CRED_LIMIT"]
    tranfile="data/train_factors1.csv"
    predfile="data/test_factors.csv"
    traindata,trainlabel,preddata=deal_data(tranfile,predfile,predictors,16000)
    # print('LogisticRegression: \r')
    # testLR(traindata, trainlabel,preddata)

    # print('GaussianNB: \r')
    # testNaiveBayes(traindata, trainlabel)
    
    # print('KNN: \r')
    # testKNN(traindata, trainlabel)
    
    # # print('SVM: \r')
    # # testSVM(traindata, trainlabel)
    
    # print('Decision Tree: \r')
    # testDecisionTree(traindata, trainlabel)
    
    print('Random Forest: \r')
    testRandomForest(traindata, trainlabel,preddata)