import numpy as np

#reading the dataset 

from numpy import genfromtxt
data = genfromtxt('bank_note_data.txt', delimiter=',')
labels = data[:,4]
features = data[:,0:4]
X = features
y = labels

#Standardizing the Data

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
from sklearn.preprocessing import MinMaxScaler
scaler_object = MinMaxScaler()
scaler_object.fit(X_train)
scaled_X_train = scaler_object.transform(X_train)
scaled_X_test = scaler_object.transform(X_test)

## Building the Network with Keras

from keras.models import Sequential
from keras.layers import Dense
# Creates model
model = Sequential()
# 8 Neurons, expects input of 4 features. 
model.add(Dense(4, input_dim=4, activation='relu'))
# Add another Densely Connected layer (every neuron connected to every neuron in the next layer)
model.add(Dense(8, activation='relu'))
# Last layer simple sigmoid function to output 0 or 1 (our label)
model.add(Dense(1, activation='sigmoid'))

### Compile Model

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
## Fit (Train) the Model

model.fit(scaled_X_train,y_train,epochs=50, verbose=2)
## Predicting New Unseen Data
model.predict_classes(scaled_X_test)
# Evaluating Model Performance

model.evaluate(x=scaled_X_test,y=y_test)
from sklearn.metrics import confusion_matrix,classification_report
predictions = model.predict_classes(scaled_X_test)
confusion_matrix(y_test,predictions)
print(classification_report(y_test,predictions))

## Saving and Loading Models

model.save('myfirstmodel.h5')
from keras.models import load_model
newmodel = load_model('myfirstmodel.h5')
newmodel.predict_classes(X_test)