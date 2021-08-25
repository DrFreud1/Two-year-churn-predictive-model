import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import cross_val_score, GridSearchCV

def boxplots(accdf, f1df, recalldf):
        
    #10-folds Shuffle Split Accuraccy Bloxplots
    plt.subplot(3, 1, 1)
    sns.boxplot(data=accdf.iloc[:,0:8], width=0.8, orient = 'v', palette= 'bone')
    plt.title('Prediction of a client dropping KS before 2 years or not')
    plt.ylabel('Accuraccy')

    #10-folds Shuffle Split F1-Score Bloxplot
    fig= plt.subplot(3, 1, 2)
    sns.boxplot(data=f1df.iloc[:,0:8], width=0.8, orient = 'v', palette= 'gist_yarg')
    plt.xlabel('Classifiers')
    plt.ylabel('F1-Score')
    
    fig= plt.subplot(3, 1, 3)
    sns.boxplot(data=recalldf.iloc[:,0:8], width=0.8, orient = 'v', palette= 'gist_yarg')
    plt.xlabel('Classifiers')
    plt.ylabel('Recall')

    plt.show()
    figure = fig.get_figure() 


def cross_vald(clf, cv, X, y):
    accuracy = cross_val_score(clf, X, np.ravel(y), cv = cv)
    f1_scr = cross_val_score(clf, X, np.ravel(y), cv = cv , scoring = 'f1')
    precision = cross_val_score(clf, X, np.ravel(y), cv = cv , scoring = 'precision')
    recall = cross_val_score(clf, X, np.ravel(y), cv = cv , scoring = 'recall')
    print("Accuracy: %0.3f (+/- %0.3f)" % (accuracy.mean(), accuracy.std() * 2))
    print("F1-Score: %0.3f (+/- %0.3f)" % (f1_scr.mean(), f1_scr.std() * 2))
    print("Precision: %0.3f (+/- %0.3f)" % (precision.mean(), precision.std() * 2))
    print("Recall: %0.3f (+/- %0.3f)" % (recall.mean(), recall.std() * 2))
    return accuracy, f1_scr, recall#, precision, recall

def find_best_param(clf, cv, X, y, param_space, scoring):
    clf = GridSearchCV(clf, param_space, n_jobs=-1, cv=cv, scoring = scoring)
    clf.fit(X, np.ravel(y))
    return clf

def print_best_param(clf, mean_thld, std_thld):
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        if (std * 2) < std_thld and mean > mean_thld:
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))