from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits as load_data
import scikitplot as skplt
#
import numpy as np
from sklearn.metrics import roc_auc_score

trainData = pd.read_csv("TrainPortion_Data.csv")
labelData = pd.read_csv("TrainPortion_Label_.csv")
testData = pd.read_excel("test-credit.xlsx")
testData = testData.columns[testData.isnull().mean() < 0.8]

trainData = trainData.drop(columns=['annual_inc_joint'])
trainData = trainData.drop(columns=['dti_joint'])
trainData = trainData.drop(columns=['desc'])

# testData = testData.drop(columns=['annual_inc_joint'])
# testData = testData.drop(columns=['dti_joint'])
# testData = testData.drop(columns=['desc'])
print(trainData.isnull().sum())

# print (trainData.isnull().mean())
# print(trainData.emp_title.mode())
# print(trainData.emp_title.value_counts())
# print(trainData['emp_title'].mode()[0])

########################## emp_title ##############################
print("emp_title row 15 before mean:", trainData.emp_title[15])  # control NaN

trainData.emp_title.fillna((trainData.emp_title.mode()[0]), inplace=True)
print("emp_title row 15 after mean:", trainData.emp_title[15])

emp_title = trainData['emp_title'].unique().tolist()
emp_title_mapping = dict(zip(emp_title, range(len(emp_title))))
trainData['emp_title'] = trainData['emp_title'].map(emp_title_mapping)

print("emp_title row 15 after mapping:", trainData.emp_title[15], "\n")

########################## title ###################################
print("title row 5 before mean:", trainData.title[5])
trainData.title.fillna((trainData.title.mode()[0]), inplace=True)
print("title row 5 after mean:", trainData.title[5])

title = trainData['title'].unique().tolist()
title_mapping = dict(zip(title, range(len(title))))
trainData['title'] = trainData['title'].map(title_mapping)

print("title row 5 after mapping:", trainData.title[5], "\n")

########################## emp_length ##############################
print("emp_length row 15 before mean:", trainData.emp_length[15])
trainData.emp_length.fillna((trainData.emp_length.mode()[0]), inplace=True)
# control if nan = teacher
print("emp_length row 15 after mean:", trainData.emp_length[15])

emp_length = trainData['emp_length'].unique().tolist()
emp_length_mapping = dict(zip(emp_length, range(len(emp_length))))
trainData['emp_length'] = trainData['emp_length'].map(emp_length_mapping)
print("emp_length row 15 after mapping:",
      trainData.emp_length[15], "\n")  # control if nan = teacher

########################## dti ##############################
print("dti row 462 before mean:", trainData.dti[462])
trainData.dti.fillna((trainData.dti.mean()), inplace=True)
print("dti row 462 after mean:", trainData.dti[462], "\n")

########################## inq_last_6mths ##############################
print("inq_last_6mths row 160159 before mean:",
      trainData.inq_last_6mths[160159])
trainData.inq_last_6mths.fillna(
    (trainData.inq_last_6mths.mean()), inplace=True)
print("inq_last_6mths row 160159 after mean:",
      trainData.inq_last_6mths[160159], "\n")

########################## total_bal_il ##############################
print("total_bal_il row 6 before mean:", trainData.total_bal_il[6])
trainData.total_bal_il.fillna((trainData.total_bal_il.mean()), inplace=True)
print("total_bal_il row 6 after mean:", trainData.total_bal_il[6], "\n")

########################## il_util ##############################
print("il_util row 6 before mean:", trainData.il_util[6])
trainData.il_util.fillna((trainData.il_util.mean()), inplace=True)
print("il_util row 6 after mean:", trainData.il_util[6], "\n")

########################## max_bal_bc ##############################
print("max_bal_bc row 6 before mean:", trainData.max_bal_bc[6])
trainData.max_bal_bc.fillna((trainData.max_bal_bc.mean()), inplace=True)
print("max_bal_bc row 6 after mean:", trainData.max_bal_bc[6], "\n")

########################## all_util ##############################
print("all_util row 6 before mean:", trainData.all_util[6])
trainData.all_util.fillna((trainData.all_util.mean()), inplace=True)
print("all_util row 6 after mean:", trainData.all_util[6], "\n")

########################## inq_fi ##############################
print("inq_fi row 6 before mean:", trainData.inq_fi[6])
trainData.inq_fi.fillna((trainData.inq_fi.mean()), inplace=True)
print("inq_fi row 6 after mean:", trainData.inq_fi[6], "\n")

########################## total_cu_tl ##############################
print("total_cu_tl row 6 before mean:", trainData.total_cu_tl[6])
trainData.total_cu_tl.fillna((trainData.total_cu_tl.mean()), inplace=True)
print("total_cu_tl row 6 after mean:", trainData.total_cu_tl[6], "\n")

########################## inq_last_12m ##############################
print("inq_last_12m row 6 before mean:", trainData.inq_last_12m[6])
trainData.inq_last_12m.fillna((trainData.inq_last_12m.mean()), inplace=True)
print("inq_last_12m row 6 after mean:", trainData.inq_last_12m[6], "\n")

########################## term ##############################
trainData['term'] = trainData['term'].str.extract('(\d+)').astype(int)

########################## home_ownership ##############################
home_ownership = trainData['home_ownership'].unique().tolist()
home_ownership_mapping = dict(zip(home_ownership, range(len(home_ownership))))
trainData['home_ownership'] = trainData['home_ownership'].map(
    home_ownership_mapping)

########################## verification_status ##############################
verification_status = trainData['verification_status'].unique().tolist()
verification_status_mapping = dict(
    zip(verification_status, range(len(verification_status))))
trainData['verification_status'] = trainData['verification_status'].map(
    verification_status_mapping)

########################## purpose ##############################
purpose = trainData['purpose'].unique().tolist()
purpose_mapping = dict(zip(purpose, range(len(purpose))))
trainData['purpose'] = trainData['purpose'].map(purpose_mapping)

########################## zip_code ##############################
trainData['zip_code'] = trainData['zip_code'].str.extract('(\d+)').astype(int)

########################## addr_state ##############################
addr_state = trainData['addr_state'].unique().tolist()
addr_state_mapping = dict(zip(addr_state, range(len(addr_state))))
trainData['addr_state'] = trainData['addr_state'].map(addr_state_mapping)

########################## earliest_cr_line ##############################
trainData['earliest_cr_line'] = trainData['earliest_cr_line'].str.extract('(\d+)').astype(int)

########################## initial_list_status ##############################
initial_list_status = trainData['initial_list_status'].unique().tolist()
initial_list_status_mapping = dict(
    zip(initial_list_status, range(len(initial_list_status))))
trainData['initial_list_status'] = trainData['initial_list_status'].map(
    initial_list_status_mapping)

########################## application_type ##############################
application_type = trainData['application_type'].unique().tolist()
application_type_mapping = dict(
    zip(application_type, range(len(application_type))))
trainData['application_type'] = trainData['application_type'].map(
    application_type_mapping)

# for column in trainData.columns:
#       print(trainData[column].head())

############################################################################################################
# validation part

# split X and y into training and testing sets
ShuffledTrainData, ShuffledLabelData = shuffle(
    trainData, labelData, random_state = 0)

x_train, x_test, y_train, y_test = train_test_split(
    ShuffledTrainData, ShuffledLabelData, test_size=0.25, random_state=0)

logreg = LogisticRegression()
y_score = logreg.fit(x_train, y_train).decision_function(x_test)

print(accuracy_score(y_test, logreg.predict(x_test)))
print(accuracy_score(y_test, logreg.predict(x_test)))


# plot
probas = logreg.predict_proba(x_test)
# skplt.metrics.plot_roc(y_true=y_test, y_probas=probas)
# plt.show()
print(roc_auc_score(y_test, y_score))

