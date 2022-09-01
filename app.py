

import os
from copyreg import pickle
from flask import Flask, render_template, request,send_from_directory
#importer le framework flask ettemplate de html
from tensorflow.keras.utils import load_img

from tensorflow.keras.utils  import img_to_array
from keras.applications.vgg16 import preprocess_input

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import pickle
app= Flask(__name__)

## save model!

STATIC_FOLDER = "static"
UPLOAD_FOLDER='C:/flaskProject/demoapp/images'

# Load model
model = tf.keras.models.load_model(STATIC_FOLDER + "/Melanoma.h5")

IMAGE_SIZE = 224

def load_and_preprocess_image():
    test_dir='C:/flaskProject/demoapp/DermMel/DermMel/test'
    test_datagen=ImageDataGenerator(rescale=1/255.0)
  
    test_generator=test_datagen.flow_from_directory(
    test_dir,target_size=(224,224),
                                                   batch_size=32,class_mode='categorical'
)


    
    return test_generator
    '''''
# Predict & classify image
def classify(model):
    batch_size = 1
    test_generator  = load_and_preprocess_image()
    prob = model.predict_generator(test_generator, steps=len(test_generator)/batch_size)
    labels = {0: 'Just another beauty mark', 1: 'Get that mole checked out'}
    label = labels[1] if prob[0][0] >= 0.5 else labels[0]
    classified_prob = prob[0][0] if prob[0][0] >= 0.5 else 1 - prob[0][0]
    return label, classified_prob'''


@app.route('/', methods=['GET']) #localhost
def home(): 
    return render_template('index.html')


@app.route('/',methods=['POST'])

def upload_file():


        
        
    imagefile= request.files['imagefile']
    image_path="C:/flaskProject/demoapp/images/" + imagefile.filename
    imagefile.save(image_path)
    image= load_img(image_path,target_size=(224,224))
    image= img_to_array(image)
    image= image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
    image=preprocess_input(image)
    prob=model.predict(image) 
    train_dir='C:/flaskProject/demoapp/DermMel/DermMel/train_sep'
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
    train_generator= train_datagen.flow_from_directory(train_dir,target_size=(224,224),
                                                   batch_size=32,class_mode='categorical'
                                                  )
    print(prob)

    print(train_generator.class_indices )
    labels = {0: 'Melanoma', 1: 'NotMelaoma'}
    label = labels[1] if prob[0][0] >= 0.5 else labels[0]
    
    print(label)
    
    
   

    return render_template(
        "index.html", image_file_name=imagefile.filename,prediction=label
    )
@app.route("/index/<filename>")

def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

    

if __name__ =='__main__':
    app.run(port=3000 ,debug=True)

  
 
'''''
 
def predict():
    imagefile= request.files['imagefile']
    image_path="C:/flaskProject/demoapp/images/" + imagefile.filename
    imagefile.save(image_path)
    image= load_img(image_path,target_size=(224,224))
    image= img_to_array(image)
    image= image.reshape((1, image.shape[0], image.shape[1],image.shape[2]))
    image=preprocess_input(image)
    yhat=model.predict(image) 

    


    #label=decode_predictions(yhat)
    #label=label[0][0]
   # classification = '%s (%.2f%%)' % (label[1], label[2]*100)

return render_template('index.html', prediction=classification)'''