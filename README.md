# Skin-cancer-Diseases-Classification-using-Transfer-Learning-and-Flask
In this work, we propose a modified InceptionV3 model for the classification of skin cancer. We propose to use Transfer learning which is a common technique for deep learning that uses pre-trained models as VGG16 ,Inception as a starting point to speed up training and to enhance the deep learning model's performance in order to classify skin cancer with a better accuracy value compared to the state of the art.
This is the source code for a skin cancer detection web app which has been implemented with flask framework a. The model has been built using deep learning library. The classifier has been trained using Kaggle dataset which contains 2 classes: melanoma and Not Melanoma.
More details about this datasets are given below:
The data is divided into:
training data (~2000 images)
validation dataset (~150 images)
test dataset (~600 images)
you can download the data from here:
https://www.kaggle.com/datasets/wanderdust/skin-lesion-analysis-toward-melanoma-detection
There are three directories: train, valid, and test. Each directory contains two subdirectories:
melanoma,NotMelanoma. Each of these directories contains images for that specific class.
The classifier has been built with transfer learning technique using a pretrained VGG16 model and InceptionV3. The final classifer achieved an accuracy of 89.2% and a F1-score of 88% on validation data. You can check out the jupyter notebook that goes along to follow all the steps which have been taken to build the model.
