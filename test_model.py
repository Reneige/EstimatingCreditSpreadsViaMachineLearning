# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 16:58:57 2023

@author: Renea
"""


from keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBRegressor, plot_tree
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error
import lime
from lime import lime_tabular

#SeniorityType	SeniorityTypeShortDescription
#SR	Senior Unsecured
#SRSEC	Senior Secured
#MTG	Senior Secured - Mortgage
#UN	Unsecured
#SEC	Secured
#SUB	Subordinated Unsecured
#1STMTG	Senior Secured - First Mortgage


# Here we use numpy function loadtxt to load the CSV file. In the file, each row corresponds to one example, with nine columns. 
# We can then split them into the input features (X) and output labels (y).

# load the dataset
#dataset1 = pd.read_excel('training_set_feb12.xlsx')
dataset1 = pd.read_excel(r'C:\Users\Renea\EstimatingCreditSpreadsViaMachineLearning\Results April 8 2023\Final Training-Testing Data Columns Y-AU.xlsx')
#dataset1 = dataset1.dropna(0) 
# split into input X and output y variables
dataset= dataset1.to_numpy()

# capture feature names
col_names = dataset1.columns.tolist()
col_names.pop()

X = dataset[:,24:46]
y = dataset[:,46]


# testing effects of standardizing
#X = dataset[:,0:20]
#x1 = dataset[:,20:25]
#y = dataset[:,25]

# standardize
#mean_train = X.mean(axis=0)
#std_train = X.std(axis=0)
#X = (X-mean_train)/std_train

#a = pd.DataFrame(X)
#b = pd.DataFrame(x1)
# add coumns one by one
#a[21]= b[0]


#Let's print the dimension of X. You may also want to have a look of a few samples.
print(X.shape)
for i in range(5):
    print(dataset[i])

X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')

import seaborn as sns

data = pd.read_excel(r"C:\Users\Renea\EstimatingCreditSpreadsViaMachineLearning\Results April7-2023\Results.xlsx")
data2 =  data[['ACTUAL','NN PREDICTED', 'BRT PREDICTED']]
plot = sns.pairplot(data2, height = 2.5)
plot = sns.relplot(data2, x='ACTUAL',y='BRT PREDICTED')

plot = sns.displot(data2, x="NN PREDICTED")

data = dataset1.iloc[:,24:47]
cols = data.columns.tolist()
cols.pop()
'''
#THE Below generates scatter plots of all the above DFs
for xvar in cols:
    print(xvar)
    df = data[[xvar,'Calculated_ZSpread']]
#    plot = sns.pairplot(df, height = 2.5)
 #   plot = sns.scatterplot(df)
    plot = sns.relplot(df, x=xvar,y='Calculated_ZSpread')
    plot.savefig(xvar+".png")
'''




"""   
# split them into training set and test set by using the train_test_split( ) method from sklearn. 
# test_size = 0.2 means 20% examples are used for test.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)


"""
2. Prepare the data
Usually there is a pre-processing step before building the model. What pre-processing should be carried out depends on the data 
and any requirement for the model. For this dataset, the eight features are in variant ranges and scales, which may lead to 
unstable weight learning. In this case, the common practice is to standardize the input data, to make each feature to be 
mean 0 and unit variance. The network would then be trained on a more stable distribution of inputs.
"""



"""
3. Build your network

"""

# define a keras model of a MLP network with three Dense layers
neural_network_model = Sequential()

neural_network_model.add(Dense(576, input_dim=22, activation = 'relu'))
neural_network_model.add(Dense(288, activation = 'relu'))

neural_network_model.add(Dense(144, activation = 'relu'))
#neural_network_model.add(BatchNormalization())
neural_network_model.add(Dense(72, activation = 'relu'))
#neural_network_model.add(BatchNormalization())
neural_network_model.add(Dense(36, activation = 'relu'))
#neural_network_model.add(BatchNormalization())
neural_network_model.add(Dense(18, activation='relu'))
#neural_network_model.add(BatchNormalization())
neural_network_model.add(Dense(9, activation='relu'))
#neural_network_model.add(BatchNormalization())
neural_network_model.add(Dense(3, activation='relu'))
#neural_network_model.add(BatchNormalization())
neural_network_model.add(Dense(1,activation='linear'))

#displays model info
neural_network_model.summary()


"""
4. Compile the model

Here we are using the mean squared error as a loss function and a mean absolute error as performance metric

we area also using the adaptive momentum learning rate optimizer ADAM

"""

neural_network_model.compile(loss='mse', optimizer=Adam(), metrics=[MeanAbsoluteError()])


"""
5. Train and validate the model
Now the model is ready to be trained. For this we simply call its fit() method. We'll train the model for 800 epochs 
in mini-batches of 128 samples. 

We also split 25% in the training set as a validation set (this is optional). Keras will measure the loss and the accuracy on 
this validation set at the end of each epoch, which is very useful to see how well the model performs. 
"""

#es_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True)

# train the model = batch size refers to mini batches for training
history = neural_network_model.fit(X_train,y_train, epochs=100, batch_size=128, validation_split=0.25)






def create_model():

    neural_network_model = Sequential()
    
    neural_network_model.add(Dense(576, input_dim=22, activation = 'relu'))
    neural_network_model.add(Dense(288, activation = 'relu'))
    
    neural_network_model.add(Dense(144, activation = 'relu'))
    #neural_network_model.add(BatchNormalization())
    neural_network_model.add(Dense(72, activation = 'relu'))
    #neural_network_model.add(BatchNormalization())
    neural_network_model.add(Dense(36, activation = 'relu'))
    #neural_network_model.add(BatchNormalization())
    neural_network_model.add(Dense(18, activation='relu'))
    #neural_network_model.add(BatchNormalization())
    neural_network_model.add(Dense(9, activation='relu'))
    #neural_network_model.add(BatchNormalization())
    neural_network_model.add(Dense(3, activation='relu'))
    #neural_network_model.add(BatchNormalization())
    neural_network_model.add(Dense(1,activation='linear'))
    neural_network_model.compile(loss='mse', optimizer=Adam(), metrics=[MeanAbsoluteError()])
    return neural_network_model







from keras.wrappers.scikit_learn import KerasClassifier, KerasRegressor
import eli5
from eli5.sklearn import PermutationImportance



my_model = KerasRegressor(build_fn=create_model, epochs=800, batch_size=128)    
my_model.fit(X_train,y_train)
perm = PermutationImportance(my_model, random_state=1).fit(X_test,y_test)


html_obj = eli5.show_weights(perm, feature_names = col_names2, top=100)

# Write html object to a file (adjust file path; Windows path is used here)
with open('NN_eli_permutations_test_data.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

"""
Note that the call to model.fit() returns a history object. This history object contains a member history, which is a 
dictionary containing the loss and extra metrics it monitored during the training process. Let's have a look.
"""
history.history.keys()


"""
below plots of the learning curves, one for loss and one for accuracy 
"""

a=history.history['loss']
b=history.history['val_loss']
c=history.history['mean_absolute_error']
d = history.history['val_mean_absolute_error']

fig = plt.figure()
fig.add_subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.legend()
plt.grid(True)
#plt.xlim([0,120]) # second var should be number of epochs
plt.ylim([0,10000])
plt.xlabel('epoch')

fig.add_subplot(1,2,2)
plt.plot(history.history['mean_absolute_error'], label='mae')
plt.plot(history.history['val_mean_absolute_error'], label='val mae')
plt.legend()
plt.grid(True)
#plt.xlim([0,120])
plt.ylim([0,60])
plt.xlabel('epoch')

"""
5. Evaluate the model

Once you are satisfied with your model's validation accuracy, you may evaluate the performance on the test set to 
estimate how the model generalize to new data. You can easily do this using the evaluate() method. The first output of 
evaluate() method is the loss of the model, and the second is the accuracy of the model.

"""

# evaluate the model on both the test set



_,accuracy = neural_network_model.evaluate(X_test,y_test)
print('Accuracy on the test set: %.2f', (accuracy))

"""
You could also use the predict() method to make predictions on new instances. The predict() outputs the likelihood of the diabetes. 
You need get the output class (0 or 1) by thresholding the probability with 0.5.

"""




y_predict = neural_network_model.predict(X_test)
for i in range(10):
    print('\n%s => %d (expected %d)' % ((X_test[i]).tolist(), y_predict[i], y_test[i]))


train_avg = np.average(y_train)
train_std = np.std(y_train)

test_avg = np.average(y_test)
test_std = np.std(y_test)

# note the y_predict here is a matrix (4723,1) and I need it as an array (4723,) or it does a matrix calculation rather than
# and element-wise calculation, so use .flatten() here
accuracy_stdev = (y_predict.flatten() - y_test).std()


print(f"\nThe Average of the train and test Z-Spreads are {train_avg:.2f} and {test_avg:.2f} respectively")
print(f"\nThe Standard Deviation of the train and test Z-Spreads are {train_std:.2f} and {test_std:.2f} respectively")
print(f"\nThe Neural Network model predicted the z-spread with an mean absolute accruacy of {accuracy:.2f}")
print(f"\nThe Neural Network model predicted the z-spread with a st-dev of {accuracy_stdev:.2f}")



''' to save and load nn model use the below'''
#neural_network_model.save('./9-layer-576-top_model')

''' load'''
neural_network_model = load_model('./Results/9-layer-576-top_model')
#


''' NOW lets employ a boosted regression tree model'''


boosted_regression_tree_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, eval_metric='mae')

# cross validation with SKLearn - Repeats K-Fold n times with different randomisation in each repetition.
#cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

#scores = cross_val_score(boosted_regression_tree_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
#print('Mean Error: %.3f St-Dev (%.3f)' % (scores.mean(), scores.std()) )

boosted_regression_tree_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test,y_test)])

y_predict_brt = boosted_regression_tree_model.predict(X_test)

# get the model R2 ((y_true - y_pred)** 2).sum() for the XGB results. Note: built-in function uses X_test vs y_test.
brt_r2 = boosted_regression_tree_model.score(X_test, y_test)

print("Mean Absolute Error: " + str(mean_absolute_error(y_predict_brt, y_test)))
print("St-Dev of error: " + str((y_predict_brt - y_test).std()))
print(f"The model's estimation coefficient of determination (R2 - R Squared) is: {brt_r2}")


#pl = plot_tree(boosted_regression_tree_model)


# plot training results - requires setting eval_set in model fitting
results = boosted_regression_tree_model.evals_result()

fig.add_subplot(2,2,2)
plt.plot(results['validation_0']['mae'], label='brt_train_mae')
plt.plot(results['validation_1']['mae'], label='brt_val_mae')
plt.legend()
plt.grid(True)
plt.xlabel('epoch')

# use eli5 to show the weights
from eli5 import show_weights, show_prediction
import webbrowser


html_obj = show_weights(boosted_regression_tree_model, feature_names=col_names, top=100)

# Write html object to a file (adjust file path; Windows path is used here)
with open('xgboost_weights.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = r'xgboost_weights.htm'
webbrowser.open(url, new=2)


html_obj_pred = show_prediction(boosted_regression_tree_model, X_test[116], show_feature_values=True, feature_names=col_names)
html_obj_pred = show_prediction(boosted_regression_tree_model, X_test[116], show_feature_values=True)

# Write html object to a file (adjust file path; Windows path is used here)
with open('xgboost_pred_X_test_116.htm','wb') as f:
    f.write(html_obj_pred.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url2 = r'xgboost_pred_X_test_116.htm'
webbrowser.open(url2, new=2)


# save xgboost model
#boosted_regression_tree_model.save_model('apr1_brt_model.json')

#Load model
#boosted_regression_tree_model = xgb.XGBRegressor()
#boosted_regression_tree_model.load_model('./Results/apr1_brt_model.json')

import xgboost
test = r"C:\Users\Renea\Documents\University of Bath\13. Dissertation\EstimatingCreditSpreadsViaMachineLearning\Results\apr1_brt_model.json"
brt = xgboost.XGBRegressor()
brt.load_model(test)
brt.n_features_in_

''' note once we override the feature_names we break the above show_prediction function so save model first'''
# push feature_names into get_booster
boosted_regression_tree_model.get_booster().feature_names = col_names

# shows a tree - there are n trees defined as n_estimators
booster = boosted_regression_tree_model.get_booster()
print(booster.get_dump()[999])

# returns occurrences of the features in splits. 
# If you divide these occurrences by their sum, you'll get Item 1. 
# Except here, features with 0 importance will be excluded.
feature_importance = boosted_regression_tree_model.get_booster().get_score(importance_type='weight')
feature_importance = boosted_regression_tree_model.get_booster().get_fscore() # same thing

# plots the values of the above feature importance, i.e. the number of occurrences in splits.
feature_importance_plt = xgb.plot_importance(boosted_regression_tree_model)



# Create a Lime explainer - more info here: https://www.kaggle.com/code/mirceavaman/explain-your-model-predictions-with-lime
explainer = lime_tabular.LimeTabularExplainer(X_train, 
                                              feature_names=col_names,
                                              class_names=['ZSpread'],
                                              verbose=True, 
                                              mode='regression')


observation=116

exp = explainer.explain_instance(X_test[observation], neural_network_model.predict, num_features=25)
html = exp.as_html(labels=None, predict_proba=True, show_predicted_value=True)
exp.save_to_file(f'nm_pred_X_test_{observation}.html', labels=None, predict_proba=True, show_predicted_value=True)
exp.as_list()
plo = exp.as_pyplot_figure()



'''
# linux only
#import tensorflow-estimator.GradientBoostedTreesModel
#model = tensorflow.estimator.BoostedTreesRegressor(feature_columns = feature_list, n_batches_per_layer = 1)

'''
