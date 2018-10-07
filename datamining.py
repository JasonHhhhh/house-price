from sklearn import linear_model
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

# load dataset
with open('case2_training.csv') as f:
    data=pd.read_csv(f,header=0)
print(data)

X, y = data.iloc[:, 1 :-1], data.iloc[:, -1]
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1)

pca=PCA(n_components=3)
x_train_pca=pca.fit_transform(X_train)
x_test_pca=pca.fit_transform(X_test)
def models_test(model, x_train, y_train, x_test, y_test, model_id):
    model.fit(x_train_pca, y_train)
    y_pred = model.predict_proba(x_test)[:, 1]
    fpr, tpr, threshold = roc_curve(y_test, y_pred, pos_label=1)
    auc_score = auc(fpr, tpr)
    print ("The auc score of model %s is %.4f" % (model_id, auc_score))

    return model

# simple logistic regression with l2 penalty
simple_regr = linear_model.LogisticRegression()
#simple_model = models_test(simple_regr, X_train, Y_train, X_test, Y_test, 1)
simple_model = models_test(simple_regr, x_test_pca, Y_train, x_test_pca, Y_test, 11)

# using l1 penalty - l1 performs better than l2 here
lasso_regr = linear_model.LogisticRegression(penalty='l1')
lasso_model = models_test(lasso_regr, X_train, Y_train, X_test, Y_test, 2)

# scale data
def normalize_data(data):
    data_norm = (data - data.mean()) / (data.max() - data.min())

    return data_norm

def normalize_price(data):
    data['Price'] = data['Price'] - data['Price'].mean() / (data['Price'].max() - data['Price'].min())

    return data

simple_normalize = models_test(simple_regr, normalize_data(X_train), Y_train, normalize_data(X_test), Y_test, 3)
lasso_normalize = models_test(lasso_regr, normalize_data(X_train), Y_train, normalize_data(X_test), Y_test, 4)
lasso_normalize_price = models_test(lasso_regr, normalize_price(X_train), Y_train, normalize_price(X_test), Y_test, 6)

# noticed the sample train data is not balanced, the # of 0s and the # of 1s are approximately 3:1
class_balance_regr = linear_model.LogisticRegression(penalty='l1', class_weight='balanced')
balanced_model = models_test(class_balance_regr, X_train, Y_train, X_test, Y_test, 5)

# grid search C, will use c = 10
C = [0.01, 10, 20, 100]
i = 6
for c in C:
    i += 1
    tune_c_regr = linear_model.LogisticRegression(penalty='l1', class_weight='balanced', C=c)
    models_test(tune_c_regr, X_train, Y_train, X_test, Y_test, i)

# feature selection
from sklearn.feature_selection import RFE
feature_selection_regr = linear_model.LogisticRegression(penalty='l1', class_weight='balanced')
selector = RFE(feature_selection_regr, 6, step=1)
selector = models_test(selector, X_train, Y_train, X_test, Y_test, 11)
print (selector.support_)
print (selector.ranking_)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
simple_dtree = models_test(clf, X_train, Y_train, X_test, Y_test, 13)

# will use grid search for learning_rate and n_estimators
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier()
ensemble_grdbst = models_test(clf, X_train, Y_train, X_test, Y_test, 14)

# The best combination is learning_rate = 0.25, n_estimators = 180 and max_depth = 2
learning_rate = [0.25, 0.3, 0.35]
n_estimators = [120, 150, 180, 200]
for lr in learning_rate:
    for ne in n_estimators:
        clf = GradientBoostingClassifier(learning_rate=lr, max_depth=2, n_estimators=ne)
        grid_search_model = models_test(clf, X_train, Y_train, X_test, Y_test, lr*ne)

# current best result auc = 0.7943
clf = GradientBoostingClassifier(learning_rate=0.25, max_depth=2, n_estimators=180)
current_best_model = models_test(clf, X_train, Y_train, X_test, Y_test, lr*ne)

# remove outliers, doesn't improve the result
from scipy import stats
import numpy as np
new_df = pd.concat([X_train, Y_train], axis=1)
filter_X = new_df[(np.abs(stats.zscore(X_train))<3).all(axis=1)]
print (filter_X.shape)
clf = GradientBoostingClassifier(learning_rate=0.25, max_depth=2, n_estimators=180)
no_outlier_model = models_test(clf, X_train, Y_train, X_test, Y_test, lr*ne)

# rule out local anomaly observations, doesn't improve the result
# The best score for this is 0.7927
from sklearn.neighbors import LocalOutlierFactor
new_df = pd.concat([X_train, Y_train], axis=1)
n_neighbors = [400]
n_outliers = [15450, 15500, 15550]
for nn in n_neighbors:
    for no in n_outliers:
        clf = LocalOutlierFactor(n_neighbors=nn, contamination=float(no)/40000)
        y_outl = clf.fit_predict(new_df)
        gbc = GradientBoostingClassifier(learning_rate=0.25, max_depth=2, n_estimators=180)
        filter_local_outlier = models_test(gbc, new_df.iloc[:no, :-1], new_df.iloc[:no, -1], X_test, Y_test, 20)

