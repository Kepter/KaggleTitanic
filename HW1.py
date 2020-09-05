import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn

train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')    

# 'PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'

# Numeric value analysis for Q7
train.describe()

# Categorical value analysis for Q8
train.describe(include=['O'])

# Survival rate based on Pclass for Q9
train[['Pclass', 'Survived']].groupby(['Pclass']).mean()

# Survival rate based on Sex for Q10
train[["Sex", "Survived"]].groupby(['Sex']).mean()

# Plots for age vs number of survived = 1 and age vs number of survived = 0 for Q11
histogram = seaborn.FacetGrid(train, col='Survived')
histogram.map(plt.hist, 'Age', bins=50)
histogram.savefig("AgeVsSurival.png")

# Plots for age vs survived vs pclass for Q12
histogram2 = seaborn.FacetGrid(train, col='Survived', row='Pclass')
histogram2.map(plt.hist, 'Age', bins=50)
histogram2.add_legend()
histogram2.savefig("AgeVsPclassVsSurvival.png")

# Plots for Fare vs Sex Vs Embarked Vs survived for Q13
histogram3 = seaborn.FacetGrid(train, row='Embarked', col='Survived')
histogram3.map(seaborn.barplot, 'Sex', 'Fare')
histogram3.add_legend()
histogram3.savefig("FareSexEmbarkedSurvived.png")

# Drop Ticket and Cabin for Q14 and Q15
train = train.drop(['Ticket', 'Cabin'], axis=1)
test = test.drop(['Ticket', 'Cabin'], axis=1)

# Combine data and change Sex to numeric for Q16
combine = [train, test]
for data in combine:
    data['Sex'] = data['Sex'].map({'female': 1, 'male': 0}).astype(int)

# Fill in missing ages using nearest neighbor with Sex and Pclass for Q17
avg_age = np.zeros((3, 2))
for data in combine:
    # Precompute nearest age for each combination
    for pclass in range(1, 4):
        for sex in range(0, 2):
            ages = data[(data['Pclass'] == pclass) & (data['Sex'] == sex)]['Age'].dropna()
            avg_age[pclass-1][sex] = math.floor(ages.median())

    # Fill missing ages based on pclass and sex
    for pclass in range(1, 4):
        for sex in range(0, 2):
            data.loc[(data['Pclass'] == pclass) & (data['Sex'] == sex) & (data['Age'].isnull()), 'Age'] = avg_age[pclass-1][sex]

    data['Age'] = data['Age'].astype(int)

# Fill in missing embarked using most common occurence for Q18
for data in combine:
    mode = data['Embarked'].dropna().mode()[0]
    data.loc[(data['Embarked'].isnull()), 'Embarked'] = mode
    data['Embarked'] = data['Embarked'].astype(str)

# Fill in missing fare value with most commmon for Q19
for data in combine:
    data.loc[(data['Fare'].isnull()), 'Fare'] = data['Fare'].dropna().mode()[0]
    data['Fare'] = data['Fare'].astype(float)

# Put Fare into bands for Q20
for data in combine:
    data.loc[(data['Fare'] <= 7.91), 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare'] = 2
    data.loc[(data['Fare'] > 31), 'Fare'] = 3

    data['Fare'] = data['Fare'].astype(int)