from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pandas as  pd


def random_forest(x_train, y_train, x_test):
    #Process
    text_clf_rf = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf-rf', RandomForestRegressor(n_estimators = 500, random_state = 42))])

    #Train
    text_clf_rf.fit(x_train, y_train);

    #Evaluate
    y_pred = text_clf_rf.predict(x_test)

    return y_pred

def svm(x_train, y_train, x_test):
    #Process
    text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                            ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=50000, random_state=42))])

    #Train
    text_clf_svm = text_clf_svm.fit(x_train, y_train)

    #Evaluate
    y_pred = text_clf_svm.predict(x_test)

    return y_pred

def evaluate(pred, true):
    print(confusion_matrix(true, pred.round()))  
    print(classification_report(true, pred.round()))  
    print(accuracy_score(true, pred.round()))
    #print(pd.crosstab(y_test, y_pred, rownames=['True'], colnames=['Predicted'], margins=True))
    print('**********************************')

def load_data():
    dev = load_files('dev', shuffle=True)
    train = load_files('train', shuffle=True)
    test = load_files('test', shuffle=True)

    x_train = train.data
    y_train = train.target
    x_test = dev.data
    y_test = dev.target

    return x_train, y_train, x_test, y_test


x_train, y_train, x_test, y_test = load_data()

print('***RANDOM FOREST***')
random_forest_pred = random_forest(x_train, y_train, x_test)
evaluate(random_forest_pred, y_test)

print('***SUPPORT VECTOR MACHINE***')
svm_pred = svm(x_train, y_train, x_test)
evaluate(svm_pred, y_test)