# Logistic Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Importing the dataset
dataset = pd.read_csv('Social_Network_Ads.csv')

#Linearity Assumption
sns.regplot(x= 'Age', y= 'Purchased', data= dataset, logistic= True).set_title("Age Log Odds Linear Plot")
sns.regplot(x= 'EstimatedSalary', y= 'Purchased', data= dataset, logistic= True).set_title("Age Log Odds Linear Plot")


sns.heatmap(dataset.isnull(),yticklabels=False,cbar=False,cmap='viridis')


X = dataset.iloc[:, [1, 2, 3]]
y = dataset.iloc[:, -1]


# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X.iloc[:, 0] = labelencoder_X.fit_transform(X.iloc[:, 0])

#Make dummy variables
onehotencoder = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding the Dummy Variable Trap
X = X[:, 1:]

'''
# Encoding the Dependent Variable
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)'''





# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, 1:] = sc.fit_transform(X[:, 1:])

#Outliners
sns.boxplot(data= X).set_title("Outlier Box Plot")

linearity_check_df = pd.concat([pd.DataFrame(X),y],axis=1) 
linearity_check_df.columns = 'Male Age Salary Purchased'.split()


sns.regplot(x= 'Age', y= 'Purchased', data= linearity_check_df, logistic= True).set_title("Age Log Odds Linear Plot")
sns.regplot(x= 'Salary', y= 'Purchased', data= linearity_check_df, logistic= True).set_title("Salary Log Odds Linear Plot")
sns.regplot(x= 'Male', y= 'Purchased', data= linearity_check_df, logistic= True).set_title("Gender Log Odds Linear Plot")


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)




# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, y_train)




#Find relevant features
from sklearn.feature_selection import RFE

rfe = RFE(classifier, 3, step=1)
rfe = rfe.fit(X, y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)






from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV

# The "accuracy" scoring is proportional to the number of correct
# classifications
rfecv = RFECV(estimator=classifier, step=1, cv=StratifiedKFold(2),
              scoring='accuracy')
rfecv.fit(X, y)

print("Optimal number of features : %d" % rfecv.n_features_)

# Plot number of features VS. cross-validation scores
plt.figure()
plt.xlabel("Number of features selected")
plt.ylabel("Cross validation score (nb of correct classifications)")
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.show()






# Predicting the Test set results
y_pred = classifier.predict(X_test)






# K-Fold cross validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10)
model_accuracy = accuracies.mean()
model_standard_deviation = accuracies.std()






# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)




#Genarate Reports
import statsmodels.api as sm

X_set = pd.DataFrame(X, columns='Male Age Salary'.split())
logit_model=sm.Logit(y,X_set.loc[:, ['Male', 'Age', 'Salary']])
result=logit_model.fit()
print(result.summary2())


# GETTING THE ODDS RATIOS, Z-VALUE, AND 95% CI
model_odds = pd.DataFrame(np.exp(result.params), columns= ['OR'])
model_odds['z-value']= result.pvalues
model_odds[['2.5%', '97.5%']] = np.exp(result.conf_int())


from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))





#ROC Curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
area_under_curve = roc_auc_score(y_test, classifier.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, classifier.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % area_under_curve)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
#plt.savefig('Log_ROC')
plt.show()