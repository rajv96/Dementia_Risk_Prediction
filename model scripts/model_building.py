#!/usr/bin/env python
# coding: utf-8

# In[11]:


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB, GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, KFold
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, confusion_matrix, f1_score, classification_report
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler


# In[3]:


import warnings
warnings.filterwarnings("ignore")


# In[8]:


# from google.colab import drive
# drive.mount('/content/drive')


# # Non-parametric models

# In[4]:


# df_non_parametric = pd.read_csv('/content/drive/My Drive/Datasets/non_parametric.csv')

df_non_parametric = pd.read_csv('non_parametric.csv')

df_non_parametric.head()


# In[5]:


df_non_parametric.drop('Unnamed: 0', axis=1, inplace=True)
df_non_parametric.head()


# In[6]:


df_non_parametric.shape


# In[7]:


X = df_non_parametric.drop('Group', axis=1)
y = df_non_parametric['Group']


# In[9]:


X.head()


# In[10]:


y


# In[8]:


#split the data into train and test split
X_np_train, X_np_test, y_np_train, y_np_test = train_test_split(X, y, test_size = 0.3, random_state = 10)


# In[53]:


X_np_train.shape, X_np_test.shape


# In[ ]:





# ## Decision Tree Classifier Algorithm

# In[55]:


DT = DecisionTreeClassifier(random_state=10)
DT.fit(X_np_train, y_np_train)
y_DT_pred = DT.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_DT_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, DT.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_DT_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# In[56]:


#initiate number of splits/folds
kf = KFold(n_splits=3, shuffle=True, random_state=10)


# ## Decision Tree - Hyperparameter Tuning

# In[57]:


params = {
    'max_depth': range(1, 21),  
    'criterion': ['entropy', 'gini'],
         }

gs_DT = GridSearchCV(DT, param_grid=params, cv = kf, scoring = 'roc_auc', n_jobs = 1)
gs_DT.fit(X_np_train, y_np_train)

DT_tuned = gs_DT.best_estimator_
print("Decision Tree TUNED MODEL: ", DT_tuned)
scores = cross_val_score(DT_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("Decision Tree TUNED SCORES: ", scores)
print("Decision Tree Tuned Bias Error: ", 1 - np.mean(scores))
print("Decision Tree Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[58]:


#predict on test data
y_pred_test_DT = DT_tuned.predict(X_np_test)


# In[59]:


#calculate the metrics on test data

print(confusion_matrix(y_np_test, y_pred_test_DT))
print(classification_report(y_np_test, y_pred_test_DT))
print(roc_auc_score(y_np_test, DT_tuned.predict_proba(X_np_test)[:,1]))


# The decision tree is overfitting on the train data but results are better than the previous models.

# ## Random Forest Classifier Algorithm

# In[60]:


RF = RandomForestClassifier(random_state=10)

RF.fit(X_np_train, y_np_train)
y_RF_pred = RF.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_RF_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, RF.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_RF_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# ## Random Forest - Hyperparameter Tuning

# In[61]:


params = {
    'max_depth': range(1, 11), 
    'n_estimators': range(1, 101), 
    'criterion': ['entropy', 'gini']
         }

gs_RF = GridSearchCV(RF, param_grid=params, cv = kf, scoring = 'roc_auc', n_jobs = -1)
gs_RF.fit(X_np_train, y_np_train)

RF_tuned = gs_RF.best_estimator_
print("Random Forest TUNED MODEL: ", RF_tuned)
scores = cross_val_score(RF_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = -1)

print("Random Forest TUNED SCORES: ", scores)
print("Random Forest Tuned Bias Error: ", 1 - np.mean(scores))
print("Random Forest Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[62]:


#predict on the test set
y_pred_test_rf = RF_tuned.predict(X_np_test)


# In[63]:


#calculate the metrics on test data
print(roc_auc_score(y_np_test, RF_tuned.predict_proba(X_np_test)[:,1]))
print(confusion_matrix(y_np_test, y_pred_test_rf))
print(classification_report(y_np_test, y_pred_test_rf))


# We are getting roc auc score of 0.9074 on the test set.

# In[ ]:





# ## Adaboost Classifier Algorithm

# In[64]:


AB = AdaBoostClassifier(random_state=10)

AB.fit(X_np_train, y_np_train)
y_AB_pred = AB.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_AB_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, AB.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_AB_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# ## AdaBoost - Hyperparameter Tuning

# In[65]:


params = {
    'base_estimator': [DT, RF],
    'n_estimators': range(1, 120),
    'learning_rate': [0.1, 0.01, 0.001, 0.15, 0.015]
         }

gs_AB = GridSearchCV(AB, param_grid=params, cv = kf, scoring = 'roc_auc', n_jobs = -1)
gs_AB.fit(X_np_train, y_np_train)

AB_tuned = gs_AB.best_estimator_
print("Adaboost TUNED MODEL: ", AB_tuned)
scores = cross_val_score(AB_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = -1)

print("Ada Boost Classifier TUNED SCORES: ", scores)
print("Ada Boost Classifier  Tuned Bias Error: ", 1 - np.mean(scores))
print("Ada Boost Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[66]:


#predict on the test data
y_pred_ab_test = AB_tuned.predict(X_np_test)


# In[68]:


#calculate the metrics on test data

print(confusion_matrix(y_np_test, y_pred_ab_test))
print(classification_report(y_np_test, y_pred_ab_test))
print(roc_auc_score(y_np_test, AB_tuned.predict_proba(X_np_test)[:,1]))


# In[ ]:


#Best model so far


# ## Gradient Boosting Algorithm

# In[70]:


GB = GradientBoostingClassifier(random_state=10)

GB.fit(X_np_train, y_np_train)
y_GB_pred = GB.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_GB_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, GB.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_GB_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# ## Gradient Boosting - Hyperparameter Tuning

# In[71]:


params = {
    'n_estimators': range(1, 100),
    'learning_rate': [0.1, 0.01, 0.001, 0.15, 0.015]
         }

gs_GB = GridSearchCV(GradientBoostingClassifier(random_state=10), param_grid=params, cv = kf, scoring = 'roc_auc', 
                     n_jobs = 1)
gs_GB.fit(X_np_train, y_np_train)

GB_tuned = gs_GB.best_estimator_
print("Gradient Boosting TUNED MODEL: ", GB_tuned)
scores = cross_val_score(GB_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("Gradient Boosting Classifier TUNED SCORES: ", scores)
print("Gradient Boosting Classifier Tuned Bias Error: ", 1 - np.mean(scores))
print("Gradient Boosting Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[72]:


#predict on test data
y_pred_test_gbm = GB_tuned.predict(X_np_test)


# In[73]:


#calculate the metrics on test data

print(confusion_matrix(y_np_test, y_pred_test_gbm))
print(classification_report(y_np_test, y_pred_test_gbm))
print(roc_auc_score(y_np_test, GB_tuned.predict_proba(X_np_test)[:,1]))


# In[ ]:





# ## XG Boost Algorithm

# In[74]:


XGB = XGBClassifier(random_state=10)

XGB.fit(X_np_train, y_np_train)
y_XGB_pred = XGB.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_XGB_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, XGB.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_XGB_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# ## XG Boost - Hyperparameter Tuning

# In[75]:


params = {
    'n_estimators': range(1, 100),
    'learning_rate': [0.1, 0.01, 0.001, 0.15, 0.015]
         }

gs_XGB = GridSearchCV(XGBClassifier(random_state=10), param_grid=params, cv = kf, scoring = 'roc_auc', n_jobs = -1)
gs_XGB.fit(X_np_train, y_np_train)

XGB_tuned = gs_XGB.best_estimator_
print("XG Boost TUNED MODEL: ", XGB_tuned)
scores = cross_val_score(XGB_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("XG Boost Classifier TUNED SCORES: ", scores)
print("XG Boost Classifier Tuned Bias Error: ", 1 - np.mean(scores))
print("XG Boost Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[82]:


#predict on test data
y_pred_test_xgb = XGB_tuned.predict(X_np_test)


# In[83]:


#calculate the metrics on test data

print(confusion_matrix(y_np_test, y_pred_test_xgb))
print(classification_report(y_np_test, y_pred_test_xgb))
print(roc_auc_score(y_np_test, XGB_tuned.predict_proba(X_np_test)[:,1]))


# ## Light GBM Algorithm

# In[78]:


LGBM = LGBMClassifier(random_state=10)
LGBM.fit(X_np_train, y_np_train)
y_LGBM_pred = LGBM.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_LGBM_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, LGBM.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_LGBM_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# ## Light GBM - Hyperparameter Tuning

# In[80]:


params = {
    'n_estimators': range(1, 100),
    'learning_rate': [0.1, 0.01, 0.001, 0.15, 0.015]
         }

gs_LGBM = GridSearchCV(LGBMClassifier(random_state=10), param_grid=params, cv = kf, scoring = 'roc_auc', n_jobs = 1)
gs_LGBM.fit(X_np_train, y_np_train)

LGBM_tuned = gs_LGBM.best_estimator_
print("LGBM TUNED MODEL: ", LGBM_tuned)
scores = cross_val_score(LGBM_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("LGBM Classifier TUNED SCORES: ", scores)
print("LGBM Boost Classifier Tuned Bias Error: ", 1 - np.mean(scores))
print("LGBM Boost Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[84]:


#predict on test data
y_pred_test_lgbm = LGBM_tuned.predict(X_np_test)


# In[85]:


#calculate the metrics on test data

print(confusion_matrix(y_np_test, y_pred_test_lgbm))
print(classification_report(y_np_test, y_pred_test_lgbm))
print(roc_auc_score(y_np_test, LGBM_tuned.predict_proba(X_np_test)[:,1]))


# In[ ]:





# ## Catboost Algorithm

# In[86]:


CB = CatBoostClassifier(random_state=10)
CB.fit(X_np_train, y_np_train)
y_CB_pred = CB.predict(X_np_test)

print("Accuracy Score: ", accuracy_score(y_np_test, y_CB_pred))
print("ROC AUC Score: ", roc_auc_score(y_np_test, CB.predict_proba(X_np_test)[:,1]))

cm = confusion_matrix(y_np_test, y_CB_pred)
sns.heatmap(cm, annot = True, fmt = 'd')
plt.show()


# ## Catboost - Hyperparameter Tuning

# In[87]:


params = {
    'n_estimators': range(1, 100),
    'learning_rate': [0.1, 0.01, 0.001, 0.15, 0.015]
         }

gs_CB = GridSearchCV(CatBoostClassifier(random_state=10), param_grid=params, cv = kf, scoring = 'roc_auc', n_jobs = 1)
gs_CB.fit(X_np_train, y_np_train)

CB_tuned = gs_CB.best_estimator_
print("Catboost TUNED MODEL: ", CB_tuned)
scores = cross_val_score(CB_tuned, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("Catboost Classifier TUNED SCORES: ", scores)
print("Catboost Classifier Tuned Bias Error: ", 1 - np.mean(scores) - np.std(scores, ddof = 1))
print("Catboost Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[88]:


#predict results on test set
y_pred_test_cb = CB_tuned.predict(X_np_test)


# In[89]:


#calculate the metrics on test data

print(confusion_matrix(y_np_test, y_pred_test_cb))
print(classification_report(y_np_test, y_pred_test_cb))
print(roc_auc_score(y_np_test, CB_tuned.predict_proba(X_np_test)[:,1]))


# Logistic - 0.71<br>
# Naive Bayes - 0.70<br>
# KNN - 0.69<br>
# KNN Tuned - 0.79<br>
# Decision Tree - 0.80<br>
# Decision Tree Tuned - 83<br>
# Random Forest 0.81 <br>
# Random Forest Tuned - 0.87<br>
# Adaboost - 0.729<br>
# Adaboost Tuned - 0.91<br>
# Gradient Boost 0.77<br>
# Gradient Boost Tuned - 0.89<br>
# XG boost - 0.77<br>
# XG Boost Tuned - 0.89<br>
# LG boost - 0.79<br>
# LG Boost Tuned - 0.89<br>
# Catboost - 0.79<br>
# Catboost Tuned - 0.88<br>

# In[53]:


df = {
    "Logistic": [0.71, 0.21, 0.073],
    "Naive Bayes": [0.70, 0.21, 0.081],
    "KNN": [0.69, 0.22, .07],
    "KNN Tuned": [0.79, 0.1, 0.11],
    "Decision Tree": [0.80, 0.09, 0.10],
    "Decision Tree Tuned": [0.83, 0.15, 0.02],
    "Random Forest": [0.81, 0.13, 0.05],
    "Random Forest Tuned": [0.87, 0.10, 0.03],
    "Ada Boost": [0.72, 0.18, 0.08],
    "Ada Boost Tuned": [0.91, 0.08, 0.01],
    "Gradient Boost": [0.77, 0.15, 0.07],
    "Gradient Boost Tuned": [0.89, 0.09, 0.02],
    "XG Boost": [0.77, 0.16, 0.06],
    "XG Boost Tuned": [0.89, 0.09, 0.02],
    "Light GBM": [0.79, 0.13, 0.07],
    "Light GBM Tuned": [0.9, 0.07, 0.02],
    "Cat Boost": [0.79, 0.1474, 0.05],
    "Cat Boost Tuned": [0.88, 0.1, 0.018]
}


# In[54]:


models = pd.DataFrame.from_dict(df, orient = 'index', columns=["ROC AUC", "Bias Error", "Variance Error"])

models


# In[55]:


models.sort_values(by = "ROC AUC", axis = 0, ascending = False)


# In[ ]:


#implementing voting and stacking classifiers


# In[50]:


voting = VotingClassifier(estimators = [('AdaBoost Tuned',AB_tuned),
                                         ('Light GBM Tuned', LGBM_tuned), 
                                         ('Gradient Boost Tuned',GB_tuned)],voting='soft')

scores = cross_val_score(voting, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("Voting Classifier TUNED ROC SCORE: ", np.mean(scores))
print("Voting Classifier Tuned Bias Error: ", 1 - np.mean(scores) - np.std(scores, ddof = 1))
print("Voting Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[51]:


stacked = StackingClassifier(estimators = [('AdaBoost Tuned',AB_tuned),
                                         ('Light GBM Tuned', LGBM_tuned), 
                                         ('Gradient Boost Tuned',GB_tuned)])

scores = cross_val_score(stacked, X_np_train, y_np_train, cv = kf, scoring = 'roc_auc', n_jobs = 1)

print("Stacked Classifier TUNED ROC SCORE: ", np.mean(scores))
print("Stacked Classifier Tuned Bias Error: ", 1 - np.mean(scores) - np.std(scores, ddof = 1))
print("Stacked Classifier Tuned Variance Error: ", np.std(scores, ddof = 1))


# In[56]:


## We can see that Adaboost with Random Forest is working best with the given data.


# In[90]:


#calculate the youden's index for optimal threshold and then plot the roc_curve for Adaboost tuned model


# In[91]:


#plot ROC curve


# In[126]:


fpr, tpr, thresholds = roc_curve(y_np_test, AB_tuned.predict_proba(X_np_test)[:,1])
plt.figure(figsize=(15,6))
plt.plot(fpr, tpr)

# set limits for x and y axes
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])

# plot the straight line showing worst prediction for the model
plt.plot([0, 1], [0, 1],'r--')

plt.title('ROC curve for Dementia Risk Prediction (Best Model)', fontsize = 12)
plt.xlabel('False positive rate (1-Specificity)', fontsize = 10)
plt.ylabel('True positive rate (Sensitivity)', fontsize = 10)

plt.text(x = 0.02, y = 0.9, s = ('AUC Score:', round(roc_auc_score(y_np_test, AB_tuned.predict_proba(X_np_test)[:,1]),4)))
                               
plt.grid(True)


# In[99]:


#Identify the best cut off value for the model


# The performance measures that we obtained above, are for the cut_off = 0.5. Now, let us consider a list of values as cut-off and calculate the different performance measures.

# We'll use the Youden's Index method for determing the optimal cut off probability

# In[100]:


#create the required dataframe
youdens_table = pd.DataFrame({'TPR': tpr,
                             'FPR': fpr,
                             'Threshold': thresholds})

#calculate the difference between TPR and FPR for each threshold
youdens_table['Difference'] = youdens_table.TPR - youdens_table.FPR

#sort the values
youdens_table = youdens_table.sort_values('Difference', ascending = False).reset_index(drop = True)

youdens_table.head()

