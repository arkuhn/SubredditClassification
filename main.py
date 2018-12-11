from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from keras.wrappers.scikit_learn import KerasRegressor
import pandas as  pd
import numpy as np
import matplotlib.pyplot as plt
import lstm


def evaluate(pred, true):
    print(confusion_matrix(true, pred.round()))  
    print(classification_report(true, pred.round()))  
    print(accuracy_score(true, pred.round()))
    fpr, tpr, thresholds = roc_curve(true, pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate or (1 - Specifity)')
    plt.ylabel('True Positive Rate or (Sensitivity)')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()
    #print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('**********************************')

def random_forest(x_train, y_train, x_test, y_test):
    #Process
    text_clf_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf-rf', RandomForestRegressor(n_estimators = 50))])

    #Train
    text_clf_rf.fit(x_train, y_train);

    #Evaluate
    y_pred = text_clf_rf.predict(x_test)
    evaluate(y_pred, y_test)

def svm(x_train, y_train, x_test, y_test):
    #Process
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=50000))])

    #Train
    text_clf_svm = text_clf_svm.fit(x_train, y_train)

    #Evaluate
    y_pred = text_clf_svm.predict(x_test)
    evaluate(y_pred, y_test)

def lregression(x_train, y_train, x_test, y_test):
    #Process
    text_clf_lr = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf-lr', LogisticRegression())])
                            
    #Train
    text_clf_lr = text_clf_lr.fit(x_train, y_train)

    #Evaluate
    y_pred = text_clf_lr.predict(x_test)
    evaluate(y_pred, y_test)

def deep_lstm(x_train, y_train, x_test, y_test):
    #Process
    clf = KerasRegressor(build_fn=lstm.create_model,verbose=0)
    text_clf_kr = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf-kr', clf)])
    
    #Train
    text_clf_kr = text_clf_kr.fit(x_train, y_train)

    #Evaluate
    y_pred = text_clf_kr.predict(x_test)
    y_pred = np.argmax(y_pred, axis=1)
    evaluate(y_pred, y_test)

def load_data():
    dev = load_files('dev', shuffle=True)
    train = load_files('train', shuffle=True)
    test = load_files('test', shuffle=True)

    x_train = train.data
    y_train = train.target
    x_test = dev.data
    y_test = dev.target
    #input_dim = (np.array(x_train)).shape[0]
    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()
print('***RANDOM FOREST***')
random_forest(x_train, y_train, x_test, y_test)

print('***SUPPORT VECTOR MACHINE***')
svm(x_train, y_train, x_test, y_test)

print('***LINEAR REGRESSION***')
lregression(x_train, y_train, x_test, y_test)

print('***LSTM***')
deep_lstm(x_train, y_train, x_test, y_test)
