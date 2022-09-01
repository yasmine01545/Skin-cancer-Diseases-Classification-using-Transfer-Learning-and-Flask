#importer les packages

import os  
         #pour manipuler les chemins
import numpy as np

from glob import glob  #pour extraire le nombre de classe
from tensorflow.keras.applications.inception_v3 import InceptionV3
#from keras.applications.vgg16 import VGG16              #importer le model
from keras.preprocessing.image import ImageDataGenerator# pour la preparation des images(augmenter data)
from keras.layers import Dense, Flatten ,Input ,Lambda                  #importer des couches quand va l'ajouter a la sortie du model pour l'addapter a notre probleme
from keras.models import Model
import tensorflow as tf




#specifier le chemin des bases train et test 
data_path='C:/flaskProject/demoapp/DermMel'#le chemin de la base
train_dir='C:/flaskProject/demoapp/DermMel/DermMel/train_sep'

test_dir='C:/flaskProject/demoapp/DermMel/DermMel/test'

print ('total testing NotMelanoma images :',len(os.listdir("c:/flaskProject/demoapp/DermMel/DermMel/test/NotMelanoma")))
print ('total trainning Melanoma images :', len(os.listdir('C:/flaskProject/demoapp/DermMel/DermMel/train_sep/Melanoma')))
print ('total trainning NotMelanoma images :',len(os.listdir('C:/flaskProject/demoapp/DermMel/DermMel/train_sep/NotMelanoma')))
print('total validation Melanoma images :',len(os.listdir('C:/flaskProject/demoapp/DermMel/DermMel/valid/Melanoma')))
print ('total valdation NotMelanoma images :', len(os.listdir('C:/flaskProject/demoapp/DermMel/DermMel/valid/NotMelanoma')))


print ('total testing Melanoma images :', len(os.listdir('C:/flaskProject/demoapp/DermMel/DermMel/test/Melanoma' )))
### afficher nos images pour mieux comprendre notre base de donnes



 #creation du reseau:preparer le model de base
IMG_SHAPE=[224,224] #pour les reseau cnn 
base_model=InceptionV3(input_shape= IMG_SHAPE + [3],include_top=False,weights='imagenet')#je vais rajouter d'autre couche a la sortie
base_model.summary()


 #bloquer le model de base
for layer in base_model.layers:
  layer.trainable=False

#ajouter les couches de sorties
x=Flatten()( base_model.output) #on rend la matrice lineaire
prediction= Dense(2,activation='softmax')(x)


#creation du model global (la connexion)
model=Model(inputs= base_model.input,outputs=prediction)
model.summary() 

from keras import optimizers

model.compile(loss="categorical_crossentropy",
              optimizer='adam',metrics=['accuracy'])


from keras.applications.imagenet_utils import preprocess_input
#from keras.applications.vgg16 import preprocess_input

#creer les photos auglmentées
#data augmentation c'est l'une des methode pour resoudre l'overfitting du modele
train_datagen=ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rescale=1/255.0,#normalise
    rotation_range=40,#on va faire une rotation de l'image en choissisent un nombre aleatoire entre 0et 40
    width_shift_range=0.2,#on va deplacer l'image de 20% de sa largeur 
    height_shift_range=0.2,
    shear_range=0.2,#on va cisailler l'image de 20 %
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
test_datagen=ImageDataGenerator(rescale=1/255.0)

#les donnes de validation just pour valider il ne faut pas faire des traitemnets a part la normalisation

train_generator= train_datagen.flow_from_directory(train_dir,target_size=(224,224),
                                                   batch_size=32,class_mode='categorical'
                                                  )
test_generator=test_datagen.flow_from_directory(
    test_dir,target_size=(224,224),
                                                   batch_size=32,class_mode='categorical'
)
  
#appliquer le modele

    

history = model.fit_generator(train_generator,
                  validation_data=test_generator
                  ,epochs=27,
                  steps_per_epoch=5
                  ,validation_steps=32,verbose=2

    
)

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

valid_loss , valid_accuracy=model.evaluate_generator(test_generator)
print("accuracy after transfer learning :{}".format(valid_accuracy))
##interpretation du model
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

import pickle
from tensorflow.keras.models import Model, load_model 
pickle.dump(model,open("model.pkl","wb"))

# Save Model and Weights
model.save('model.h5')
model.save_weights('model.hdf5')
# Load saved model
model = load_model('model.h5')



