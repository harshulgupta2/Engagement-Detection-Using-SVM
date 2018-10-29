# Engagement Detection using Open Face features

import pandas as pd
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
import sys
import seaborn
from imblearn.under_sampling import RandomUnderSampler, CondensedNearestNeighbour, EditedNearestNeighbours, RepeatedEditedNearestNeighbours
from imblearn.under_sampling import NeighbourhoodCleaningRule, TomekLinks
from imblearn.over_sampling import ADASYN,RandomOverSampler, SMOTE  
from imblearn.ensemble import BalanceCascade 
from sklearn.metrics import accuracy_score,  roc_curve, auc
from sklearn.metrics import confusion_matrix, cohen_kappa_score, roc_auc_score
from sklearn.metrics import average_precision_score, recall_score, f1_score, precision_score
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.cross_validation import StratifiedKFold
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC

importance = 0
corr = 0
gridsearch = 0

def f_importances(coef, names):
    imp = coef
    imp,names = zip(*sorted(zip(imp,names)))
    plt.barh(range(len(names)), imp, align='center')
    plt.yticks(range(len(names)), names)
    plt.show()

df = pd.read_csv('train.csv')
if importance:
    target = df.label
    df.drop(['label'], 1, inplace=True)
    clf = RandomForestClassifier(n_estimators=50)
    clf = clf.fit(df, target)
    features = pd.DataFrame()
    features['feature'] = df.columns
    features['importance'] = clf.feature_importances_
    features.sort_values(by=['importance'], ascending=True, inplace=True)
    features.set_index('feature', inplace=True)
    features.plot(kind='barh', figsize=(25, 25))
    plt.show()
    exit()

if corr:
    df_corr = df.corr()
    plt.figure(figsize=(15,10))
    seaborn.heatmap(df_corr, cmap="YlGnBu") # Displaying the Heatmap
    seaborn.set(font_scale=2,style='white')
    plt.title('Heatmap correlation')
    plt.show()
    exit()

X_train = df.as_matrix(columns = ['gaze0_x','gaze0_y','gaze0_z','gaze1_x','gaze1_y','gaze1_z','poser_x','poser_y','poser_z','au23','au05','au12'])    ## Features with High Correlation and Importance Values

train_label1 = df.as_matrix(columns = ['label'])
y_train = np.ravel(train_label1)

rus = EditedNearestNeighbours(random_state=42)
X_resampled, y_resampled = rus.fit_sample(X_train, y_train)

df1 = pd.read_csv('test.csv')
test_data = df1.as_matrix(columns = ['gaze0_x','gaze0_y','gaze0_z','gaze1_x','gaze1_y','gaze1_z','poser_x','poser_y','poser_z','au23','au05','au12'])

test_label1 = df1.as_matrix(columns = ['label'])
test_label = np.ravel(test_label1)								

if gridsearch:
	C_range = 10. ** np.arange(-2, 3)
	gamma_range = 10. ** np.arange(-3, 2)
	param_grid = dict(gamma=gamma_range, C=C_range)
	grid = GridSearchCV(SVC(), param_grid=param_grid, cv=StratifiedKFold(y=y_resampled, n_folds=5))
	grid.fit(X_resampled, y_resampled)
	print("The best classifier is: ", grid.best_estimator_)
	exit()

clf = svm.SVC(C=175.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape=None, degree=3, gamma=0.031, kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)

clf.fit(X_resampled, y_resampled)    
out_label = clf.predict(test_data)
out_prob = clf.predict_proba(test_data)
fpr, tpr, _ = roc_curve(test_label, out_label)
roc_auc = auc(fpr, tpr)
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

#from sklearn.metrics import roc_auc_score
#r_a_score = roc_auc_score(test_label, out_label)
#print("ROC-AUC-Score:", r_a_score)

acc_fraction = accuracy_score(test_label, out_label)
correct_class = accuracy_score(test_label, out_label, normalize = False)
conf_matrix = confusion_matrix(test_label, out_label)
avg_precn_score = average_precision_score(test_label, out_label, average = 'weighted')
recall = recall_score(test_label, out_label)
f_score = f1_score(test_label, out_label)
kappa_score = cohen_kappa_score(test_label, out_label)
auc_area = roc_auc_score(test_label, out_label)
precision = precision_score(test_label, out_label)
print('EditedNearestNeighbours')
print('Accuracy::', acc_fraction)
print('No of correctly clasified::', correct_class)
print('Confusion matrix ::', conf_matrix)
print('Average precision score::', avg_precn_score)
print('Recall::', recall)
print('F1 score::', f_score)
print('Cohens kappa::', kappa_score)
print('AUC value::', auc_area)
print('Precision', precision)