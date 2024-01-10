#Importing libraries
import numpy as np
import cv2
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Flatten
import matplotlib.pyplot as plt

#Loading ready data set 
X_train = np.loadtxt('input.csv', delimiter = ',')
Y_train = np.loadtxt('labels.csv', delimiter = ',')
X_test = np.loadtxt('input_test.csv', delimiter = ',')
Y_test = np.loadtxt('labels_test.csv', delimiter = ',')

#Reshaping The data set to fit the model we made dim will be 100 *100 and 3 will be for colors 
X_train = X_train.reshape(len(X_train), 100, 100, 3)
Y_train = Y_train.reshape(len(Y_train), 1)
X_test = X_test.reshape(len(X_test), 100, 100, 3)
Y_test = Y_test.reshape(len(Y_test), 1)

#Rescaling Values
X_train = X_train/255.0
X_test = X_test/255.0

#Building Empty model
model = Sequential()

#adding conv layer consist of 32 filter dim(3,3) with ativation Func Relu Takes matrix 100*100*3
model.add(Conv2D(32, (3,3), activation = 'relu', input_shape = (100, 100, 3)))

#Max pooling with  filter (2,2)
model.add(MaxPooling2D((2,2)))

#adding conv layer consist of 32 filter dim(3,3) with ativation Func Relu
model.add(Conv2D(32, (3,3), activation = 'relu'))

#Max pooling with  filter (2,2)
model.add(MaxPooling2D((2,2)))

#flatten the Matrix into Array 
model.add(Flatten())

#adding Layer of ANN with 64 neurons and (Relu) Active function
model.add(Dense(64, activation = 'relu'))

#adding The Out Put  Layer of ANN with 1 neuron with sigmoid because its binary classifcation (Betweeen cat or Dog)
model.add(Dense(1, activation = 'sigmoid'))

#Backprobagation (loss Func(binary_crossentropy) bec of binary model, Optimizer (adam),train on accuracy )
model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
#Fitting the model with our Data
#epochs nuber of cycle and size of data we take each we should dosent increase that to not over fitting 
model.fit(X_train, Y_train, epochs = 15, batch_size = 64)
#Evaluate the Model
model.evaluate(X_test, Y_test)

#Function take the  path for an image on Pc and resize it to fit model then Predict the output
def Predict_image(Pc_Location):
    #takes the input and change signs
    Location = Pc_Location.replace("\\", "/")
    Location = Pc_Location.replace("\"", "")
    #takes the locationg to read the image 
    image = cv2.imread(Location)
    #conver The image from BGR TO RGB (to be colored)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #reshape image to be as the model
    image_resized = cv2.resize(image_rgb, (100, 100), fx = 1, fy = 1)
    #showing the image
    plt.imshow(image_resized)
    plt.axis('off')
    plt.show()
    #Predicting

    y_pred = model.predict(image_resized.reshape(1, 100, 100, 3))
    y_pred = y_pred > 0.5
    if(y_pred == 0):
     pred = 'dog'
    else:
     pred = 'cat'
    print("Our model says it is a :", pred)

# To Try serval input then close program manualy 
while True:
  
    Predict_image( input(" Enter path ")) 
