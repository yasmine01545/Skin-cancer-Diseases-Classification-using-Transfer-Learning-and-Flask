
# Skin-cancer-Diseases-Classification-using-Transfer-Learning-and-Flask

Melanoma Cancer is a dangerous form of skin-cancer. Though, it is a rare form of skin-cancer but it is highly fatal. In this repo, we are training a Deep CNN for finding out if a lesion is cancerous or not. 
We use a pre-trained Inception Model to generate as a feature extractor since our dataset is relatively small.

This article explains the approach and the results in detail.



## Organization

The rest of the paper is organised as follows:


```bash
Section 1 details materials and proposed method.
```

```bash
Section 2 deploy the model using flask.
```


In this work, we propose a modified InceptionV3 model for the classification of skin cancer. We propose to use Transfer learning which is a common technique for deep learning that uses pre-trained models as VGG16 ,Inception as a starting point to speed up training and to enhance the deep learning model's performance in order to classify skin cancer with a better accuracy value compared to the state of the art.
## DATA

The model has been built using deep learning library. The classifier has been trained using Kaggle dataset which contains 2 classes: melanoma and Not Melanoma.
More details about this datasets are given below:
The data is divided into:
## training data (~2000 images)
## validation dataset (~150 images)
## test dataset (~600 images)
you can download the data from here:
https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection
There are three directories: train, valid, and test. Each directory contains two subdirectories:
melanoma,NotMelanoma. Each of these directories contains images for that specific class.


## 

- Sample images of Melanoma from the dataset:


![AUG_0_354](https://user-images.githubusercontent.com/80918787/187921737-e8ab1669-7361-4dce-8d86-5ab1ed0e9816.jpeg)
![AUG_0_1721](https://user-images.githubusercontent.com/80918787/187921986-ef1635e2-36db-4367-9583-7089b48cac66.jpeg)



- Sample images of NotMelanoma from the dataset:

![ISIC_0024497](https://user-images.githubusercontent.com/80918787/187922213-ec301530-bcb1-4101-9f7b-5764d548701d.jpg)


## build with
-TensorFlow

-Keras

-Python

-flask

## Deploying ML Model using Flask
This is a simple project to elaborate how to deploy a Machine Learning model using Flask API

## Prerequisites
You must have Scikit Learn, Pandas (for Machine Leraning Model) and Flask (for API) installed.

Flask version: 0.12.2 conda install flask=0.12.2 (or) pip install Flask==0.12.2

## Project Structure:
This project has four major parts :

- model.py - This contains code fot our Machine Learning model to predict if the image is melanoma or not.
- app.py - This contains Flask APIs that receives employee details through GUI or API calls, computes the precited value based on our model and returns it.
- template - This folder contains the HTML template (index.html) to allow user to enter patient detail and displays the predicted image.
- static - This folder contains the css folder which has the styling required for out index.html file.





Run app.py using below command to start Flask API

```bash
python app.py
```
![image](https://user-images.githubusercontent.com/80918787/195153926-92851e01-2af3-4e88-8c72-7754fee01cc1.png)
The home page of the application is shown above, Once you upload the image a preview will be 
shown.

![image](https://user-images.githubusercontent.com/80918787/195154260-818c6928-36ad-4db6-a421-af2d278ad348.png)
The preview of the uploaded image is shown as a confirmation and you can make the predictions 
from there as shown 

![image](https://user-images.githubusercontent.com/80918787/195154503-0b7e4d56-f3d4-4190-8d11-e4fde6ca927b.png)
Now the image is taken into the flask framework where the prediction is made and the result is 
shown on the same page as shown .

