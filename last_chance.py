#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile as zf
files = zf.ZipFile("melanome_data.Zip", 'r')
files.extractall('lasttt_data')
files.close()


# In[ ]:





# In[2]:


import tensorflow as tf
import os,glob
import cv2
from tqdm.notebook import tqdm_notebook as tqdm
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Convolution2D,Dropout ,Dense,MaxPooling2D
from keras.layers import BatchNormalization
from keras.layers import MaxPooling2D
from keras.layers import Flatten


# In[3]:


data_dir='./lasttt_data/melanome_data'


# In[4]:


os.listdir(data_dir)


# In[5]:


#on va deviser notre data test and training folder
os.chdir('./lasttt_data/melanome_data/Melanoma')
Melanoma_labels=[] 


Melanoma_images=[]


for i in tqdm(os.listdir()):
    img=cv2.imread(i)
    img=cv2.resize(img,(224,224))
    Melanoma_images.append(img)
    Melanoma_labels.append('Melanoma')


# In[6]:


os.chdir('C:/Users/msi/lasttt_data/melanome_data/NotMelanoma')
NotMelanoma_labels=[] 
NotMelanoma_images=[]

for i in tqdm(os.listdir()):
    img=cv2.imread(i)
    img=cv2.resize(img,(224,224))
    NotMelanoma_images.append(img)
    NotMelanoma_labels.append('NotMelanoma')


# In[9]:


get_ipython().run_line_magic('matplotlib', 'inline')
X=np.concatenate((NotMelanoma_images,Melanoma_images))
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(1,4,i+1)
    plt.imshow(X[i],cmap="gray")
    
    plt.axis('off')
plt.show()


# In[11]:


# Split into training and testing sets for both types of images
Melanoma_x_train, Melanoma_x_test, Melanoma_y_train, Melanoma_y_test = train_test_split(
    Melanoma_images, Melanoma_labels, test_size=0.2)
NotMelanoma_x_train, NotMelanoma_x_test, NotMelanoma_y_train, NotMelanoma_y_test = train_test_split(
    NotMelanoma_images, NotMelanoma_labels, test_size=0.2)



# In[14]:


# Merge sets for both types of images
X_train = np.concatenate((NotMelanoma_x_train, Melanoma_x_train), axis=0)
X_test = np.concatenate((NotMelanoma_x_test, Melanoma_x_test), axis=0)
y_train = np.concatenate((NotMelanoma_y_train, Melanoma_y_train), axis=0)
y_test = np.concatenate((NotMelanoma_y_test, Melanoma_y_test), axis=0)


# In[16]:


from sklearn.preprocessing import LabelBinarizer

from tensorflow.keras.utils import to_categorical
# Make labels into categories - either 0 or 1, for our model
y_train = LabelBinarizer().fit_transform(y_train)
y_train = to_categorical(y_train)

y_test = LabelBinarizer().fit_transform(y_test)
y_test = to_categorical(y_test)


# In[17]:


print("X_train Shape: ", X_train.shape) 
print("X_test Shape: ", X_test.shape)
print("y_train Shape: ", y_train.shape) 
print("y_test Shape: ", y_test.shape)


# In[18]:


from keras.applications.vgg16 import VGG16              #importer le model
from keras.preprocessing.image import ImageDataGenerator# pour la preparation des images(augmenter data)
from keras.layers import Dense, Flatten ,Input ,Lambda                  #importer des couches quand va l'ajouter a la sortie du model pour l'addapter a notre probleme
from keras.models import Model


# In[19]:


#creation du reseau:preparer le model de base
IMG_SHAPE=[224,224] #pour les reseau cnn 
base_model=VGG16(input_shape= IMG_SHAPE + [3],include_top=False,weights='imagenet')#je vais rajouter d'autre couche a la sortie
base_model.summary()


# In[20]:


#bloquer le model de base
for layer in base_model.layers:
  layer.trainable=False


# In[21]:


base_model.output


# In[22]:


#ajouter les couches de sorties
x=Flatten()( base_model.output) #on rend la matrice lineaire
prediction= Dense(2,activation='softmax')(x)


# In[23]:


#creation du model global (la connexion)
model=Model(inputs= base_model.input,outputs=prediction)

model.summary() 


# In[24]:


from keras import optimizers

model.compile(loss="categorical_crossentropy",
              
              optimizer='adam',metrics=['accuracy'])


# In[25]:


history = model.fit(X_train,y_train,
                    epochs=10, 
                    validation_data=(X_test,y_test),
                    verbose = 1,
                    initial_epoch=0)


# In[26]:


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])

plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_ct_accuracy.png')
plt.show()


# In[27]:


plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')

plt.legend(['Training', 'Testing'])
plt.savefig('vgg_ct_loss.png')
plt.show()


# In[24]:


acc=history.history['accuracy']
val_acc =history.history['val_accuracy']#validation acc
loss = history.history['loss']
val_loss= history.history['val_loss']
epochs= len(acc)
plt.figure(figsize=(12,8))
plt.plot(np.arange(epochs),acc,label="train acc")

plt.plot(np.arange(epochs),val_acc,label="validation acc")
plt.title("train acc vs validation acc")
plt.legend()
plt.show()
plt.figure(figsize=(12,8))
plt.plot(np.arange(epochs),loss,label="train loss")

plt.plot(np.arange(epochs),val_loss,label="validation loss")
plt.title("train loss vs validation loss")
plt.legend()
plt.show() 
#this is overfitting 


# In[28]:


y_pred = model.predict(X_test, batch_size=32)


# In[29]:


# Convert to Binary classes
y_pred_bin = np.argmax(y_pred, axis=1)
y_test_bin = np.argmax(y_test, axis=1)


# In[30]:



from sklearn.metrics import confusion_matrix, roc_curve
import seaborn as sns
def plot_confusion_matrix(normalize):
  classes = ['Melanoma','NotMelanoma ']
  tick_marks = [0.5,1.5]
  cn = confusion_matrix(y_test_bin, y_pred_bin,normalize=normalize)
  sns.heatmap(cn,cmap='Reds',annot=True)
  plt.xticks(tick_marks, classes)
  plt.yticks(tick_marks, classes)
  plt.title('Confusion Matrix')
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

print('Confusion Matrix without Normalization')
plot_confusion_matrix(normalize=None)

print('Confusion Matrix with Normalized Values')
plot_confusion_matrix(normalize='true')


# In[31]:


fpr, tpr, thresholds = roc_curve(y_test_bin, y_pred_bin)
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.rcParams['font.size'] = 12
plt.title('ROC curve for our model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.grid(True)


# In[32]:


from sklearn.metrics import classification_report
print(classification_report(y_test_bin, y_pred_bin))


# In[33]:



train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[34]:


model.summary()


# In[35]:


train_aug = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)


# In[36]:


### **Training the model**
epochs=10
batch_size = 32

history = model.fit(train_aug.flow(X_train, y_train, batch_size=batch_size),
                    validation_data=(X_test, y_test),
                    validation_steps=len(X_test) / batch_size,
                    steps_per_epoch=len(X_train) / batch_size,
                    epochs=epochs)


# In[37]:


acc=history.history['accuracy']
val_acc =history.history['val_accuracy']#validation acc
loss = history.history['loss']
val_loss= history.history['val_loss']
epochs= len(acc)
plt.figure(figsize=(12,8))
plt.plot(np.arange(epochs),acc,label="train acc")

plt.plot(np.arange(epochs),val_acc,label="validation acc")
plt.title("train acc vs validation acc")
plt.legend()
plt.show()
plt.figure(figsize=(12,8))
plt.plot(np.arange(epochs),loss,label="train loss")

plt.plot(np.arange(epochs),val_loss,label="validation loss")
plt.title("train loss vs validation loss")
plt.legend()
plt.show()


# In[40]:


from tensorflow.keras.models import Model, load_model 
# Save Model and Weights
model.save('modelMelanoma.h5')
model.save_weights('modelMelanoma.hdf5')
# Load saved model
model = load_model('modelMelanoma.h5')


# In[46]:


# Save Model and Weights
model.save('./Downloads/Melanoma.h5')
model.save_weights('./Melanoma.hdf5')


# In[29]:


# Load saved model
model = load_model('./Melanoma.h5')


# In[ ]:




