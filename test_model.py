# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 16:58:57 2023

@author: Renea
"""


"""
1. Load the dataset

In this tutorial, we are going to use Pima Indians Diabetes dataset which is a standard machine learning dataset from the UCI Machine 
Learning repository. The dataset can be downloaded from here. Place it in the same location as this notebook file.

This is a comparably small dataset, with 768 samples in total. There are eight input features (X) and one output variable (y).

Input features (X):

Number of times pregnant
Plasma glucose concentration a 2 hours in an oral glucose tolerance test
Diastolic blood pressure (mm Hg)
Triceps skin fold thickness (mm)
2-Hour serum insulin (mu U/ml)
Body mass index (weight in kg/(height in m)^2)
Diabetes pedigree function
Age (years)
Output Variables (y):

Class label (1 for diabetes, 0 for not)
First we need import all necessory libraries which we will need to work upon.
"""

from keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.layers import Dense, BatchNormalization, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping
from keras.metrics import MeanSquaredError, MeanAbsoluteError
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from xgboost import XGBRegressor, plot_tree
from sklearn.model_selection import RepeatedKFold, cross_val_score
from sklearn.metrics import mean_absolute_error


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
dataset1 = pd.read_excel('training_set_feb21.xlsx')
dataset1 = dataset1.fillna(0) 
# split into input X and output y variables
dataset= dataset1.to_numpy()


X = dataset[:,6:24]
y = dataset[:,24]

#Let's print the dimension of X. You may also want to have a look of a few samples.
print(X.shape)
for i in range(5):
    print(dataset[i])

X = np.asarray(X).astype('float32')
y = np.asarray(y).astype('float32')


#cols = d1.columns
#plot = sns.pairplot(d1[cols], height = 2.5)
#plt.show();
    
"""   
# Let's split them into training set and test set by using the train_test_split( ) method from sklearn. 
# test_size = 0.2 means 20% examples are used for test.
# You must treat the test set as unseen data, which means any model adjustment must be done only on the training set.
"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


"""
2. Prepare the data
Usually there is a pre-processing step before building the model. What pre-processing should be carried out depends on the data 
and any requirement for the model. For this dataset, the eight features are in variant ranges and scales, which may lead to 
unstable weight learning. In this case, the common practice is to standardize the input data, to make each feature to be 
mean 0 and unit variance. The network would then be trained on a more stable distribution of inputs.
"""

#mean_train = X_train.mean(axis=0)
#std_train = X_train.std(axis=0)
#X_train = (X_train-mean_train)/std_train
#X_test = (X_test-mean_train)/std_train # Apply the same transformation on the testing set 


"""
3. Build your network
Now we are ready to build our neural network. In Keras, we first create a sequential model, then add layers one by one.

As there are only 8 input features in this dataset, we may consider to build a small-size MLP: three fully 
connected layers (or Dense layers). For each Dense layer, we specify the number of neurons in the layer as the first 
argument, and specify the activation function using the activation argument. For the first Dense layer, we also need give the 
input feature dimension (input_dim=8).

We use the Rectified Linear Unit (ReLU) activation function for the 2 hidden layers, as ReLU usually produces better performance 
compared to the Sigmoid or Tanh functions. We use a sigmoid on the output layer to ensure the network output is between 0 and 1 
and easy to map to either a probability of class 1 or snap to a hard classification of either class with a default threshold of 0.5.

Now lets's interprete the following code as follows.

The model expects the input data containing 8 features (or columns).
We add the first hidden layer with 12 neurons and use the ReLU activation fuction.
We add the second hidden layer with 6 neurons and use the ReLu activation function.
Finally, we add the output layer with 1 neuron and use the sigmoid activation function.
"""

# define a keras model of a MLP network with three Dense layers
neural_network_model = Sequential()

neural_network_model.add(Dense(576, input_dim=18, activation = 'relu'))
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
After the model is created, you must call the compile() method to specify the loss function and the optimizer to use. 
Optionally, you can specify a list of extra metrics to compute and report during training and evaluation.

Regarding the binary classification problem, the typical loss function is binary_crossentropy defined in Keras.

We choose the efficient stochastic gradient descent algorithm adam as the optimizer. It is a popular choice as it adaptively 
chooses the learning rate and gives good results in a wide range of problems.

Since it is a classification problem, it is more intuitive to measure and report the classification accuracy, defined via the 
metrics argument.
"""

# compile the model
neural_network_model.compile(loss='mse', optimizer=Adam(), metrics=[MeanAbsoluteError()])

#neural_network_model.compile(loss='mse', optimizer=Adam(), metrics=[MeanAbsoluteError()])

"""
5. Train and validate the model
Now the model is ready to be trained. For this we simply call its fit() method. We'll train the model for 120 epochs 
(120 itrations over all samples in the training set), in mini-batches of 32 samples. (If you don't know what mini-batch means, 
don't worry, we will talk about it in our later lectures.)

We also split 25% examples in the training set as a validation set (this is optional). Keras will measure the loss and the accuracy on 
this validation set at the end of each epoch, which is very useful to see how well the model performs. Based on the performance on the 
validation set, we could modify the model and tune the hyperparameters accordingly.
"""

#es_callback = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, restore_best_weights=True)

# train the model = batch size refers to mini batches for training
history = neural_network_model.fit(X_train,y_train, epochs=800, batch_size=128, validation_split=0.25)




"""
Note that the call tomodel.fit() returns a history object. This history object contains a member history, which is a 
dictionary containing the loss and extra metrics it monitored during the training process. Let's have a look.
"""
history.history.keys()


"""
If you use this dictionary to create a panda DataFrame and call its plot() method, you will get the four learning curves 
plotted in one figure. We will plot them in two separate figures, as follows.
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
You can see that both the training loss and validation loss decrease rapidly before 25 epochs. After that the gap between training 
loss and validation loss get larger, which means the model starts to overfit. In our later lectures, we will talk about what 
overfitting is, what causes overfitting and how to prevent overfitting when training deep learning models.

In this particular case, the model looks like it performed better on the validation set than on the training set at the beginning 
of training. But that's not the case. Indeed, the validation loss is computed at the end of each epoch, while the training loss is 
computed during each epoch. So the training curve should be shifted by half an epoch to the left. If you do that, you will see the 
training and validation curves overlap almost perfectly at the beginning of training.


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


print(f"\nThe Average of the train and test Z-Spreads are {train_avg:.2f} and {test_avg:.2f} respectively")
print(f"\nThe Standard Deviation of the train and test Z-Spreads are {train_std:.2f} and {test_std:.2f} respectively")
print(f"\nThe Neural Network model predicted the z-spread with an mean absolute accruacy of {accuracy:.2f}")



''' NOW lets employ a boosted regression tree model'''


boosted_regression_tree_model = XGBRegressor(n_estimators=1000, learning_rate=0.05)

# cross validation with SKLearn - Repeats K-Fold n times with different randomisation in each repetition.
cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)

scores = cross_val_score(boosted_regression_tree_model, X_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1)
print('Mean Error: %.3f St-Dev (%.3f)' % (scores.mean(), scores.std()) )

boosted_regression_tree_model.fit(X_train, y_train)

y_predict_brt = boosted_regression_tree_model.predict(X_test)
print("Mean Absolute Error: " + str(mean_absolute_error(y_predict_brt, y_test)))

pl = plot_tree(boosted_regression_tree_model)
pl.show()

results = boosted_regression_tree_model.evals_result()


# use eli5 to show the weights
from eli5 import show_weights
import webbrowser
html_obj = show_weights(boosted_regression_tree_model)

# Write html object to a file (adjust file path; Windows path is used here)
with open('xgboost_weights.htm','wb') as f:
    f.write(html_obj.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url = r'xgboost_weights.htm'
webbrowser.open(url, new=2)

from eli5 import show_prediction
html_obj_pred = show_prediction(boosted_regression_tree_model, X_test[0], show_feature_values=True)

# Write html object to a file (adjust file path; Windows path is used here)
with open('xgboost_pred.htm','wb') as f:
    f.write(html_obj_pred.data.encode("UTF-8"))

# Open the stored HTML file on the default browser
url2 = r'xgboost_pred.htm'
webbrowser.open(url2, new=2)

# shows a tree
booster = boosted_regression_tree_model.get_booster()
print(booster.get_dump()[0])



'''
# linux only
#import tensorflow-estimator.GradientBoostedTreesModel
#model = tensorflow.estimator.BoostedTreesRegressor(feature_columns = feature_list, n_batches_per_layer = 1)

'''