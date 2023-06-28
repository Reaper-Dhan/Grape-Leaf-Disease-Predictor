from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Activation
from tensorflow.keras.utils import to_categorical
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pickle
import random
from datetime import datetime
from packaging import version

#change to the path of the training dataset
dirtrain='./grape_leaves_DB/train'

#change to the path of the testing dataset
dirtest='./grape_leaves_DB/test'

#variable that has the name of the diseases in order of the class
categories=["Black_rot","Esca_(Black_Measles)","Healthy","Leaf_blight_(Isariopsis_Leaf_Spot)"]

# just to display an image
for c in categories:
    path=os.path.join(dirtrain,c)
    for i in os.listdir(path):
        img_array=cv2.imread(os.path.join(path,i))
        #print(img_array.shape)
        plt.imshow(img_array)
        plt.show()
        break
    break

training_data = []

def create_training_data():
    count=[]
    for c in categories:
        path=os.path.join(dirtrain,c)#creating the path of each class (folder)
        class_num=categories.index(c)#label is equal to the position of the class in 'categories' variable
        c=0
        for i in os.listdir(path):
            c=c+1
            try:
                img_array=cv2.imread(os.path.join(path,i))#creating the path of each image
                training_data.append([img_array,class_num])
            except Exception as e:
                pass
        count.append(c)
    return count

count_train=create_training_data() #function called to extract images from the training folder

testing_data = []
def create_testing_data():
    count=[]
    for c in categories:
        path=os.path.join(dirtest,c)
        class_num=categories.index(c)
        c=0
        for i in os.listdir(path):
            c=c+1
            try:
                img_array=cv2.imread(os.path.join(path,i))
                #img_array=cv2.resize(img_array,(128,128))
                testing_data.append([img_array,class_num])
            except Exception as e:
                pass
        count.append(c)
    return count

count_test=create_testing_data() #function called to extract images from the testing folder

"""
print(len(training_data))
print(count_train)
print(len(testing_data))
print(count_test)
"""

#shuffling the dataset to avoid successive training on same class of images
random.shuffle(training_data)
random.shuffle(testing_data)

x_train = []
y_train = []
x_test = []
y_test = []

#separating the images and label for the model
for features, label in training_data:
    x_train.append(features)
    y_train.append(label)
x_train=np.array(x_train).reshape(-1,256,256,3)
#reshaping -1 means that the it can be any value i.e. the original value which is the no. of images
#256x256 for the dimension of the image and 3 for the the layers Red Green and Blue (RGB)

#displaying an image
x=cv2.resize(training_data[0][0],(256,256))
plt.imshow(x,cmap='gray')

#separating the images and label for evaluation
for features, label in testing_data:
    x_test.append(features)
    y_test.append(label)
x_test=np.array(x_test).reshape(-1,256,256,3)

#saving the constructed training dataset using pickle
def save_training_data(x_train,y_train):
    pickle_out=open("x_train_coloured.pickle","wb")
    pickle.dump(x_train,pickle_out)
    pickle_out.close()

    pickle_out=open("y_train_coloured.pickle","wb")
    pickle.dump(y_train,pickle_out)
    pickle_out.close
save_training_data(x_train,y_train)

#saving the constructed testing dataset using pickle
def save_testing_data(x_test,y_test):
    pickle_out=open("x_test_coloured.pickle","wb")
    pickle.dump(x_test,pickle_out)
    pickle_out.close()

    pickle_out=open("y_test_coloured.pickle","wb")
    pickle.dump(y_test,pickle_out)
    pickle_out.close()
save_testing_data(x_test,y_test)

#example of loading data from pickle file
def load_data():
    pickle_in=open("x_train_coloured.pickle","rb")
    x_train=pickle.load(pickle_in)
    return x_train
#once the pickle files are ready no need to process the images form folder again and again

#creating the neural network model
K.clear_session()
model=Sequential()
model.add(layers.Conv2D(32,(3,3),padding='same',input_shape=(256,256,3),activation='relu'))
model.add(layers.Conv2D(32,(3,3),activation='relu'))

model.add(layers.MaxPool2D(pool_size=(8,8)))

model.add(layers.Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(layers.Conv2D(32,(3,3),activation='relu'))

model.add(layers.MaxPool2D(pool_size=(8,8)))

model.add(Activation('relu'))

model.add(Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(4,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

#compiling the network using the following loss and optimizer
model.summary()

#converting the training label to categorical
y_train_cat=to_categorical(y_train,4) #4 in the no. of categories

#converting the training label to categorical
y_test_cat=to_categorical(y_test,4)

#fit the model i.e. training the model and batch size can be varies
model.fit(x_train,y_train_cat,batch_size=32,
          epochs=10,verbose=1,validation_split=0.15,shuffle=True)
#validating the model with 15% data after every epoch which is also shuffled after each epoch

#saving the trained model so that no need to fit again for next time
model.save("leaf_disease_coloured.h5")

#example of loading the saved model
new_model=tf.keras.models.load_model("leaf_disease_coloured.h5")

#evaluating the saved model
loss, acc = new_model.evaluate(x_test,y_test_cat, verbose=2)
print('Restored model, accuracy: {:5.2f}%'.format(100*acc))

'''from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())'''

#predicting a image using the model

#'d' is the path of the image
d='./sample.jpg'
img=cv2.imread(d)
#uncomment the below line if the image is not 256x256 by default
#img_array=cv2.resize(img_array,(256,256))
plt.imshow(img)

#reshaping the image to make it compatible for the argument of predict function
img=img.reshape(-1,256,256,3)

#predicting the class of the image
predict_class=new_model.predict(img)
predict_class=np.argmax(predict_class,axis=1)

#using the predict class as the index for categories defined at the beginning to display the name
print("The given image is from the category: ",categories[predict_class[0]])