#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import all necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.feature_selection import chi2,SelectKBest
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.impute import SimpleImputer


# In[2]:


#load the dataset to the notebook
college = pd.read_csv('go to college.csv')


# In[3]:


#check the first 5 rows of the data
college.head()


# In[4]:


#Check a quick statistical review of the numerical columns in the dataset
college.describe()


# In[5]:


#Check for important information about the dataset
college.info()


# In[6]:


#Check for missing values in the dataset
college.isnull().sum()


# In[7]:


#convert categorical values into numerical ones
le = LabelEncoder()
for col in college.columns:
    if college[col].dtypes == 'object' or 'bool':
        college[col] = le.fit_transform(college[col])


# In[8]:


college.head()


# In[9]:


# Explore the distribution of 'average_grades' using a histogram
plt.hist(college['average_grades'], bins=10, color='skyblue')
plt.xlabel('Average Grades')
plt.ylabel('Frequency')
plt.title('Distribution of Average Grades')
plt.show()


# In[10]:


# Create a scatterplot between 'average_grades' and 'interest'
plt.scatter(college['average_grades'], college['interest'])
plt.xlabel('Average Grades')
plt.ylabel('Interest')
plt.title('Scatterplot of Average Grades vs Interest')
plt.show()


# In[11]:


# Compute the correlation matrix
corr_matrix = college.corr()

# Plot the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Display the plot
plt.show()


# In[12]:


#Split the dataset into X and y
X = college.drop(['will_go_to_college'], axis = 1)
y = college['will_go_to_college']


# In[13]:


#view X
X


# In[14]:


#view y
y


# In[15]:


#Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X , y , test_size= 0.2, random_state= 0)


# In[16]:


#Standardize the data
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[17]:


X_train


# In[18]:


X_test


# In[19]:


#view the distribution of class labels in the training set
y_train.value_counts()


# In[20]:


#Create the logistic regression model
classifier_LR = LogisticRegression(random_state=0)


# In[21]:


classifier_LR.fit(X_train, y_train)


# In[22]:


y_pred_LR = classifier_LR.predict(X_test)
y_pred_LR


# In[23]:


#check for the accuracy of the model
acc_LR = metrics.accuracy_score(y_test, y_pred_LR)
acc_LR


# In[24]:


#check for the sensitivity of the model
recall_LR = metrics.recall_score(y_test, y_pred_LR)
recall_LR


# In[25]:


#check for the precision of the model
precision_LR = metrics.precision_score(y_test, y_pred_LR)
precision_LR


# In[26]:


#Check for the F1 score of the model
f1_score_LR =metrics.f1_score(y_test, y_pred_LR)
f1_score_LR


# In[27]:


#check for the ROC_AUC score
roc_auc_LR = metrics.roc_auc_score(y_test, y_pred_LR)
roc_auc_LR


# In[28]:


#view the confusion matrix
cm_LR = metrics.confusion_matrix(y_test, y_pred_LR)
cm_LR


# In[29]:


#view the classification report
result_LR = metrics.classification_report(y_test, y_pred_LR)
print('Classification Report:\n')
print(result_LR)


# In[30]:


#visualize the confusion matrix
ax = sns.heatmap(cm_LR, cmap = 'flare', annot= True, fmt = 'd')
plt.xlabel('Predicted Class', fontsize = 12)
plt.ylabel('True Class', fontsize = 12)
plt.title('Confusion Matrix', fontsize = 12)
plt.show()


# CHECK FOR OVERFITTING

# In[31]:


#Make predictions based on the training set
y_train_pred_LR = classifier_LR.predict(X_train)

#Make predictions based on the test set
y_test_pred_LR = classifier_LR.predict(X_test)

#Calculate the accuracy of the training set
acc_train_LR = metrics.accuracy_score(y_train, y_train_pred_LR)

#Calculate the accuracy of the test set
acc_test_LR = metrics.accuracy_score(y_test, y_test_pred_LR)

print('Training Accuracy:', acc_train_LR)
print('Test Accuracy:', acc_test_LR)


# TO APPLY THE KNN ALGORITHM, THE OPTIMAL NUMBER OF K(NEAREST NEIGHBORS) HAS TO BE FIRST DETERMINED
# 
# APPLY GRIDSEARCH CROSS-VALIDATION METHOD TO DETERMINE K

# In[32]:


#Initialize the gridsearch
k_range = list(range(1, 31))


param_grid = dict(n_neighbors = k_range)

knn = KNeighborsClassifier()
grid = GridSearchCV(knn, param_grid, cv = 10, scoring = 'accuracy')

grid.fit(X, y)


# In[33]:


#view the results of the gridsearch
results = grid.cv_results_

print(grid.best_score_)
print(grid.best_params_)


# RUN THE K-NEAREST NEIGHBOR CLASSIFIER

# In[34]:


#Create the K-nearest neighbor model
classifier_KNN = KNeighborsClassifier(n_neighbors= 21)


# In[35]:


#Train the model using the training set
classifier_KNN.fit(X_train, y_train)


# In[36]:


#use the trained model to predict outcome of the test set 
y_pred_KNN = classifier_KNN.predict(X_test)


# In[37]:


#view the predicted outcome
y_pred_KNN


# In[38]:


#Check for the accuracy of the model
acc_KNN = metrics.accuracy_score(y_test, y_pred_KNN)
acc_KNN


# In[39]:


#Check for the precision of the model
prec_KNN = metrics.precision_score(y_test, y_pred_KNN)
prec_KNN


# In[40]:


#Check for the sensitivity of the model
recall_KNN = metrics.recall_score(y_test, y_pred_KNN)
recall_KNN


# In[41]:


#Check for the F1 score of the model
f1_score_KNN =metrics.f1_score(y_test, y_pred_KNN)
f1_score_KNN


# In[42]:


#Check for the ROC_AUC score of the model
roc_auc_KNN = metrics.roc_auc_score(y_test, y_pred_KNN)
roc_auc_KNN


# In[43]:


#view the confusion matrix of the model
cm_KNN = metrics.confusion_matrix(y_test, y_pred_KNN)
cm_KNN


# In[44]:


#view the classification report
result_KNN = metrics.classification_report(y_test, y_pred_KNN)
print('Classification Report:\n')
print(result_KNN)


# In[45]:


#visualize the confusion matrix of the model
ax = sns.heatmap(cm_KNN, cmap = 'flare', annot= True, fmt = 'd')
plt.xlabel('Predicted Class', fontsize = 12)
plt.ylabel('True Class', fontsize = 12)
plt.title('Confusion Matrix', fontsize = 12)
plt.show()


# CHECK FOR OVERFITTING

# In[46]:


#Make predictions based on the training set
y_train_pred_KNN = classifier_KNN.predict(X_train)

#Make predictions based on the test set
y_test_pred_KNN = classifier_KNN.predict(X_test)

#Calculate the accuracy of the training set
acc_train_KNN = metrics.accuracy_score(y_train, y_train_pred_KNN)

#Calculate the accuracy of the test set
acc_test_KNN = metrics.accuracy_score(y_test, y_test_pred_KNN)

print('Training Accuracy:', acc_train_KNN)
print('Test Accuracy:', acc_test_KNN)


# In[ ]:




