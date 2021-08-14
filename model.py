import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential,load_model
from keras.layers import Dense,Dropout,Activation,Flatten
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import time
import joblib

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

print("Shape of Train data: ",train.shape)
print("Shape of Test data: ",test.shape)

## dividing the data into dependent and independent data
xtrain = train.drop(['subject', 'Activity', 'ActivityName'], axis=1)
ytrain = train['ActivityName']

xtest = test.drop(['subject', 'Activity', 'ActivityName'], axis=1)
ytest = test['ActivityName']
print("Done....")
sc = StandardScaler()
xtrain  = sc.fit_transform(xtrain)
xtest = sc.transform(xtest)

print('xtrain and ytrain : ({},{})'.format(xtrain.shape, ytrain.shape))
print('xtest  and ytest  : ({},{})'.format(xtest.shape, ytest.shape))

ytrain = ytrain.map({'LAYING':0, 'SITTING':1,'STANDING':2,'WALKING':3,'WALKING_DOWNSTAIRS':4,'WALKING_UPSTAIRS':5})
ytest = ytest.map({'LAYING':0, 'SITTING':1,'STANDING':2,'WALKING':3,'WALKING_DOWNSTAIRS':4,'WALKING_UPSTAIRS':5})
ytrain = to_categorical(ytrain)
ytest = to_categorical(ytest)
print(ytest.shape)
print(ytrain.shape)


model = Sequential(name="HAR-Net")
model.add(Dense(256,input_shape=(561,)))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(6,activation="softmax"))

checkpointer = ModelCheckpoint('HAR-Net.h5', save_best_only=True,monitor='val_accuracy',mode='auto')
opt = tf.keras.optimizers.Adam(learning_rate=0.01,decay=0.01)


print("Model architecture is done....")
model.compile(loss = "categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
print("Model compiled....")
print("Training is starting....")
start = time.time()
hist = model.fit(xtrain,ytrain,batch_size=16,epochs=200,validation_data=(xtest, ytest),callbacks=[checkpointer])
print("Model training is over....")
print("Total Time taken: ",time.time()-start)
print("Model saved....")

# plotting the figures
print("Plotting the figures....")
plt.figure(figsize=(15,10))
plt.plot(hist.history['accuracy'],c='b',label='train')
plt.plot(hist.history['val_accuracy'],c='r',label='validation')
plt.title("Model Accuracy vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("ACCURACY")
plt.legend(loc='lower right')
plt.savefig('./img/accuracy.png')


plt.figure(figsize=(15,10))
plt.plot(hist.history['loss'],c='orange',label='train')
plt.plot(hist.history['val_loss'],c='g',label='validation')
plt.title("Model Loss vs Epochs")
plt.xlabel("EPOCHS")
plt.ylabel("LOSS")
plt.legend(loc='upper right')
plt.savefig('./img/loss.png')
print("Figures saved in the disk....")

model=load_model("HAR-Net.h5")
# testing the model
print("Testing the model....")
print("The result obtained is...\n")
model.evaluate(xtest,ytest)

joblib.dump(sc,'scaler.pkl')
