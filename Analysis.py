import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn
import sklearn
import KNN
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot as plt

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')    

# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

train.describe()
train.describe(include=['O'])

train[['Pclass', 'Survived']].groupby(['Pclass']).mean()
train[["Sex", "Survived"]].groupby(['Sex']).mean()

histogram = seaborn.FacetGrid(train, col='Survived')
histogram.map(plt.hist, 'Age', bins=50)
histogram.savefig("AgeVsSurival.png")

histogram2 = seaborn.FacetGrid(train, col='Survived', row='Pclass')
histogram2.map(plt.hist, 'Age', bins=50)
histogram2.add_legend()
histogram2.savefig("AgeVsPclassVsSurvival.png")

histogram3 = seaborn.FacetGrid(train, row='Embarked', col='Survived')
histogram3.map(seaborn.barplot, 'Sex', 'Fare')
histogram3.add_legend()
histogram3.savefig("FareSexEmbarkedSurvived.png")

train = train.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)
test = test.drop(['Ticket', 'Cabin', 'Name', 'PassengerId'], axis=1)

combine = [train, test]
for data in combine:
    data['Gender'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)

combine[0] = combine[0].drop(['Sex'], axis=1)
combine[1] = combine[1].drop(['Sex'], axis=1)

avg_age = np.zeros((3, 2))
for data in combine:
    # Precompute nearest age for each combination
    for pclass in range(1, 4):
        for sex in range(0, 2):
            ages = data[(data['Pclass'] == pclass) & (data['Gender'] == sex)]['Age'].dropna()
            avg_age[pclass-1][sex] = math.floor(ages.median())

    # Fill missing ages based on pclass and sex
    for pclass in range(1, 4):
        for sex in range(0, 2):
            data.loc[(data['Pclass'] == pclass) & (data['Gender'] == sex) & (data['Age'].isnull()), 'Age'] = avg_age[pclass-1][sex]

    data['Age'] = data['Age'].astype(int)

for data in combine:
    mode = data['Embarked'].dropna().mode()[0]
    data.loc[(data['Embarked'].isnull()), 'Embarked'] = mode
    data['Embarked'] = data['Embarked'].astype(str)

for data in combine:
    data.loc[(data['Fare'].isnull()), 'Fare'] = data['Fare'].dropna().mode()[0]
    data['Fare'] = data['Fare'].astype(float)

for data in combine:
    data.loc[(data['Fare'] <= 7.91), 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[(data['Fare'] > 31), 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)

for data in combine:
    data['Relatives'] = data['SibSp'] + data['Parch']
    data['Relatives'] = data['Relatives'].astype(int)

    data = data.drop(['SibSp', 'Parch'], axis=1)

combine[0] = combine[0].drop(['SibSp', 'Parch'], axis=1)
combine[1] = combine[1].drop(['SibSp', 'Parch'], axis=1)

for data in combine:
    data.loc[(data['Embarked'] == 'S'), 'Embarked'] = 0
    data.loc[(data['Embarked'] == 'C'), 'Embarked'] = 1
    data.loc[(data['Embarked'] == 'Q'), 'Embarked'] = 2

for data in combine:
    data.loc[(data['Age'] <= 10), 'Age'] = 0
    data.loc[(data['Age'] > 10) & (data['Age'] <= 20), 'Age'] = 1
    data.loc[(data['Age'] > 20) & (data['Age'] <= 30), 'Age'] = 2
    data.loc[(data['Age'] > 30) & (data['Age'] <= 40), 'Age'] = 3
    data.loc[(data['Age'] > 40) & (data['Age'] <= 60), 'Age'] = 4
    data.loc[(data['Age'] > 60), 'Age'] = 5

# Prepare training data
x_train = combine[0].drop(['Survived'], axis=1)
y_train = combine[0]['Survived']

# Create decision tree based on data
dec_tree = DecisionTreeClassifier()
dec_tree.fit(x_train, y_train)

dec_tree_train_acc = dec_tree.score(x_train, y_train)

# Plot decision tree
feat_names = ['Pclass', 'Age', 'Fare', 'Embarked', 'Gender', 'Relatives']
class_names = ['Not Survived', 'Survived']
dec_tree_plot = plt.figure(figsize=(100,100))
_ = tree.plot_tree(dec_tree, feature_names=feat_names, class_names=class_names, filled=True)
dec_tree_plot.savefig("decision_tree")

# Create Random Forest based on data
rand_forest = RandomForestClassifier()
rand_forest.fit(x_train, y_train)

rand_forest_train_acc = rand_forest.score(x_train, y_train)

# Apply five-fold cross validation to decision tree
combine = combine[0]
split_size = int(len(combine) / 5)
running_acc = 0
for itr in range(5):
    # 1/5 of data as test
    x_test = combine[itr * split_size : (itr+1) * split_size]
    y_test = x_test['Survived']
    x_test = x_test.drop(['Survived'], axis=1)

    # 4/5 of data as train
    x_train = pd.concat([combine[0 : max(itr * split_size - 1, 0)], combine[(itr+1) * split_size + 1 :]])
    y_train = x_train['Survived']
    x_train = x_train.drop(['Survived'], axis=1)

    dec_tree = DecisionTreeClassifier()
    dec_tree.fit(x_train, y_train)
    running_acc = running_acc + dec_tree.score(x_test, y_test)

dec_tree_avg_accuracy = running_acc / 5

# Apply five-fold cross validation to random forest
running_acc = 0
for itr in range(5):
    # 1/5 of data as test
    x_test = combine[itr * split_size : (itr+1) * split_size]
    y_test = x_test['Survived']
    x_test = x_test.drop(['Survived'], axis=1)

    # 4/5 of data as train
    x_train = pd.concat([combine[0 : max(itr * split_size - 1, 0)], combine[(itr+1) * split_size + 1 :]])
    y_train = x_train['Survived']
    x_train = x_train.drop(['Survived'], axis=1)

    rand_forest = RandomForestClassifier()
    rand_forest.fit(x_train, y_train)
    running_acc = running_acc + rand_forest.score(x_test, y_test)

rand_forest_avg_accuracy = running_acc / 5

# Create Naive Bayes classifier and test with 5-fold cross validation
running_acc = 0
running_prec = 0
running_rec = 0
running_f1 = 0
for itr in range(5):
    # 1/5 of data as test
    x_test = combine[itr * split_size : (itr+1) * split_size]
    y_test = x_test['Survived']
    x_test = x_test.drop(['Survived'], axis=1)

    # 4/5 of data as train
    x_train = pd.concat([combine[0 : max(itr * split_size - 1, 0)], combine[(itr+1) * split_size + 1 :]])
    y_train = x_train['Survived']
    x_train = x_train.drop(['Survived'], axis=1)

    nb = GaussianNB()
    nb.fit(x_train, y_train)

    # Calculate stats
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    y_pred = nb.predict(x_test)
    y_test = y_test.tolist()
    for i in range(len(y_pred)):
        if (y_pred[i] == y_test[i]):
            if y_pred[i] == 1:
                tp += 1
            else:
                tn += 1
        else:
            if y_pred[i] == 1:
                fp += 1
            else:
                fn += 1

    acc = (tp+tn)/(tp+fp+fn+tn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = (2 * rec * prec / (rec + prec))
    print("ITR: ", itr, " ACC: ", acc, " PREC: ", prec, " RECALL: ", rec, " F1: ", f1)

    running_acc += acc
    running_prec += prec
    running_rec += rec
    running_f1 += f1

running_acc /= 5
running_prec /= 5
running_rec /= 5
running_f1 /= 5

# Create custom KNN and test with values of k from 1 to 20
custom_knn = KNN.KNN()

x_test = combine[0 : split_size]
y_test = x_test['Survived']
x_test = x_test.drop(['Survived'], axis=1)

x_train = combine[split_size+1:]
y_train = x_train['Survived']
x_train = x_train.drop(['Survived'], axis=1)

x_test = x_test.values.tolist()
y_test = y_test.values.tolist()
x_train = x_train.values.tolist()
y_train = y_train.values.tolist()

custom_knn.fit(x_train, y_train)
for k in range(40):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    for i in range(len(x_test)):
        pred = custom_knn.predict(x_test[i], k+1)

        if (pred == y_test[i]):
            if pred == 1:
                tp += 1
            else:
                tn += 1
        else:
            if pred == 1:
                fp += 1
            else:
                fn += 1
    
    acc = (tp+tn)/(tp+fp+fn+tn)
    prec = tp/(tp+fp)
    rec = tp/(tp+fn)
    f1 = (2 * rec * prec / (rec + prec))
    print("K: ", k+1, " ACC: ", acc, " PREC: ", prec, " RECALL: ", rec, " F1: ", f1)

