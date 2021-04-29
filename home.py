# --------Les imports ----------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np # linear algebra
from lime import lime_image
from bokeh.colors import color
from sklearn.metrics import roc_curve, roc_auc_score, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix
from bokeh.plotting import figure
import keras
from skimage.segmentation import mark_boundaries
from keras.layers import Dense, Dropout, Flatten, GlobalAveragePooling2D, MaxPool2D, BatchNormalization, ZeroPadding2D
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import Adam, SGD
from keras.models import Model, Sequential, load_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Flatten, Conv2D, Activation, Dense, Dropout, MaxPooling2D

import tensorflow as tf

from sklearn.model_selection import train_test_split
import cv2
import os

#-----Model de Deep Leaning------
@st.cache
def importerImages():
    labels = ['PNEUMONIA', 'NORMAL']
    img_size = 180
    def get_training_data(data_dir):
        data = []
        for label in labels:
            path = os.path.join(data_dir, label)
            class_num = labels.index(label)
            for img in os.listdir(path):
                try:
                    img_arr = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
                    resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                    data.append([resized_arr, class_num])
                except Exception as e:
                    print(e)
        return np.array(data)


    train = get_training_data('C:/Users/Administrateur/OneDrive/Formation DATASCIENCE/Projet DATA SCIENCE/chest_xray/train')
    test = get_training_data('C:/Users/Administrateur/OneDrive/Formation DATASCIENCE/Projet DATA SCIENCE/chest_xray/test')
    val = get_training_data('C:/Users/Administrateur/OneDrive/Formation DATASCIENCE/Projet DATA SCIENCE/chest_xray/val')


    IMG_SIZE = 180
    x_train = []
    y_train = []

    x_val = []
    y_val = []

    x_test = []
    y_test = []

    for feature, label in train:
        x_train.append(feature)
        y_train.append(label)

    for feature, label in val:
        x_val.append(feature)
        y_val.append(label)

    for feature, label in test:
        x_test.append(feature)
        y_test.append(label)



    x_train = np.array(x_train) / 255
    x_val = np.array(x_val) / 255
    x_test = np.array(x_test) / 255


    x_train = x_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_train = np.array(y_train)

    x_val = x_val.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_val = np.array(y_val)

    x_test = x_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y_test = np.array(y_test)

    x_global = np.concatenate((x_train,x_test),axis=0)
    y_global = np.concatenate((y_train,y_test),axis=0)



    X_train, X_test, Y_train, Y_test = train_test_split(x_global, y_global, test_size=0.33, random_state=42)
    return X_train, X_test, Y_train, Y_test

def generationModel(X_train):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), padding="same", input_shape=X_train.shape[1:]))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), padding="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(2, 2))
    #model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.add(Dropout(0.2))

    model.add(Activation("sigmoid"))
    metrics = [
      'accuracy',
      tf.keras.metrics.Precision(name='precision'),
      tf.keras.metrics.Recall(name='recall'),
    ]

def lancementModel(model,metrics,X_train, Y_train,X_test,Y_test):
    model.compile(loss='binary_crossentropy', # loss function
                  optimizer='adam',                # optimization algorithm
                  metrics=metrics)

    training_history = model.fit(X_train, Y_train,
                                 validation_split = 0.2,
                                 epochs = 25,
                                 batch_size = 200)

    scores = model.evaluate(X_test, Y_test)


def importerModel(adresse):
    reconstructed_model = keras.models.load_model(adresse)
    return reconstructed_model
def importerHistorique(adresse):
    r = pd.read_json(adresse)
    return r

def print_results(y_test, y_pred):
    try:
        st.write('Accuracy   : {:.5f}'.format(accuracy_score(y_pred , y_test)))
        st.write('AUC        : {:.5f}'.format(roc_auc_score(y_test , y_pred)))
        st.write('Precision  : {:.5f}'.format(precision_score(y_test , y_pred)))
        st.write('Recall     : {:.5f}'.format(recall_score(y_test , y_pred)))
        st.write('F1         : {:.5f}'.format(f1_score(y_test , y_pred)))
        st.write('Confusion Matrix : \n', confusion_matrix(y_test, y_pred))
    except:
        pass

def printHistorique(history,epochs):

    epochs_array = [i for i in range(epochs)]
    fig, ax = plt.subplots(1, 3)
    train_precision = history['precision']
    train_recall = history['recall']
    train_loss = history['loss']

    val_precision = history['val_precision']
    val_recall = history['val_recall']
    val_loss = history['val_loss']
    fig.set_size_inches(20, 5)

    ax[0].plot(epochs_array, train_loss, 'g-o', label='Training Loss')
    ax[0].plot(epochs_array, val_loss, 'r-o', label='Validation Loss')
    ax[0].set_title('Training & Validation Loss')
    ax[0].legend()
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].grid(True)

    ax[1].plot(epochs_array, train_precision, 'go-', label='Training Precision')
    ax[1].plot(epochs_array, val_precision, 'ro-', label='Validation Precision')
    ax[1].set_title('Training & Validation Precision')
    ax[1].legend()
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Precision')
    ax[1].grid(True)

    ax[2].plot(epochs_array, train_recall, 'go-', label='Training Recall')
    ax[2].plot(epochs_array, val_recall, 'ro-', label='Validation Recall')
    ax[2].set_title('Training & Validation Recall')
    ax[2].legend()
    ax[2].set_xlabel('Epochs')
    ax[2].set_ylabel('Recall')
    ax[2].grid(True)

    st.pyplot(fig)

def printAccuracy(history,epochs):
    epochs_array = [i for i in range(epochs)]
    train_acc = history['accuracy']
    val_acc = history['val_accuracy']

    p = figure(
        title='Précision (acc) des jeux de train et de validation.',
        x_axis_label='Epochs',
        y_axis_label='Précision (acc)')
    p.line(epochs_array, train_acc, legend_label='Précision du train (acc)', line_width=2,color='green')
    p.line(epochs_array, val_acc, legend_label='Précision de la validation (acc)', line_width=2,color='red')

    p.circle(epochs_array, train_acc, line_width=2, color='green',fill_color="white")
    p.circle(epochs_array, val_acc, line_width=2, color='red',fill_color="white")

    p.legend.location = "bottom_right"
    p.legend.click_policy = "hide"

    st.bokeh_chart(p)

def printImage(y_test,y_pred,x_test,IMG_SIZE):
    st.write('Images des erreurs de predictions du modèle')
    incorrect = np.nonzero(y_test != y_pred)[0]
    fig, ax = plt.subplots(3, 2, figsize=(15, 15))
    ax = ax.ravel()
    plt.subplots_adjust(wspace=0.25, hspace=0.75)
    plt.tight_layout()
    i = 0
    for c in incorrect[:6]:
        ax[i].set_xticks([])
        ax[i].set_yticks([])
        ax[i].imshow(x_test[c].reshape(IMG_SIZE, IMG_SIZE), cmap='gray', interpolation='none')
        ax[i].set_title('Prédiction de Classe: {}, Véritable Classe: {}'.format(y_pred[c], y_test[c]))
        i += 1
    st.pyplot(fig)
def importerImage():
    return 0

def result():
    IMG_SIZE = 180
    epochs = 45
    X_train, X_test, Y_train, Y_test = importerImages()
    model = importerModel("model_pneumonie_1_KAGGLE.h5")
    historique = importerHistorique("historique_model.json")

    predictions = model.predict(x=X_test)
    y_pred = np.round(predictions).reshape(1, -1)[0]
    print_results(Y_test, y_pred)
    printHistorique(historique,epochs)
    printAccuracy(historique, epochs)
    printImage(Y_test,y_pred,X_test,IMG_SIZE)


def pretraitementImage(uploaded_file_pred,color_on):
    file_bytes = np.asarray(bytearray(uploaded_file_pred.read()), dtype=np.uint8)
    if color_on == True:
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    else:
        opencv_image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    # st.image(opencv_image, caption='Sunrise by the mountains')
    imageResize = cv2.resize(opencv_image, (180, 180))

    st.image(imageResize, caption='Image Chargée')
    l = []
    l.append(imageResize)

    image255 = np.array(np.array(l)) / 255
    imagePredction = image255.reshape(-1, 180, 180, 1)
    return imagePredction
def predictionModel(imagePredction,model_pred):

    predictions = model_pred.predict(x=imagePredction)

    return predictions


#-----Le streamlit------


status_text = st.sidebar.empty()
add_selectbox = st.sidebar.selectbox(
    "Selectionner un mode",
    ("Détection Pneumonie","Détection Xray","Détails du modèle")
)

if add_selectbox == "Détection Pneumonie":
    st.title("Détection d'une pneumonie dans une radio des poumons d'un enfant")
    st.write(" ")
    model_pneumonie_1_KAGGLE = importerModel("model_pneumonie_1_KAGGLE.h5")
    model_detection_Xray = importerModel("model_detection_Xray.h5")
    uploaded_file = st.file_uploader("Télécharger une image", accept_multiple_files=False,type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        imagePredction = pretraitementImage(uploaded_file,False)
        #imagePredction_couleur = pretraitementImage(uploaded_file, True)
        predictions_Xray = predictionModel(imagePredction, model_detection_Xray)
        y_pred_Xray = np.round(predictions_Xray).reshape(1, -1)[0]
        #y_pred_Xray = 1
        if y_pred_Xray == 1:
            st.write("L'image est une radio des poumons d'un enfant. (_certitude de ",int(predictions_Xray*100),"%_)")
            predictions_Pneumonie = predictionModel(imagePredction, model_pneumonie_1_KAGGLE)
            y_pred = np.round(predictions_Pneumonie).reshape(1, -1)[0]
            labels = ['PNEUMONIA', 'NORMALE']
            st.write("Le modèle de Deep Learning identifie l'image comme : ","**",labels[int(y_pred)],"**")

            if y_pred == 1:
                pourcentage_prediction = int(predictions_Pneumonie*100)
            elif y_pred == 0:
                pourcentage_prediction = (1 - int(predictions_Pneumonie))* 100
            else:
                pourcentage_prediction = 0
            if pourcentage_prediction <= 95:
                pourcentage_prediction_text = "Faible"
            elif pourcentage_prediction <= 98 & pourcentage_prediction > 95:
                pourcentage_prediction_text = "Moyen"
            else:
                pourcentage_prediction_text = "Elevé"
            st.write("Le pourcentage de certitude est de : ", pourcentage_prediction,"% _(",pourcentage_prediction_text,")_")

            #explainer = lime_image.LimeImageExplainer()
            #explanation = explainer.explain_instance(imagePredction, model_pneumonie_1_KAGGLE.predict, top_labels=5, hide_color=0, num_samples=1000)
            #temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True,
                                                        #num_features=5, hide_rest=True)
            #st.pyplot(plt.imshow(mark_boundaries(temp / 2 + 0.5, mask)))

        else:
            st.write("L'image n'est pas une radio des poumons d'un enfant. (_certitude de ",(1 - int(predictions_Xray))* 100,"%_)")

elif add_selectbox == "Détection Xray":
    st.title("Détection d'une image de radio des poumons d'enfant")
    st.write(" ")
    model_detection_Xray = importerModel("model_detection_Xray.h5")
    uploaded_file = st.file_uploader("Télécharger une image", accept_multiple_files=False, type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        imagePredction = pretraitementImage(uploaded_file,False)
        predictions_Xray = predictionModel(imagePredction, model_detection_Xray)

        y_pred = np.round(predictions_Xray).reshape(1, -1)[0]
        labels = ['Autre', 'XRAY']
        st.write("Le modèle de Deep Learning identifie l'image comme : ","**",labels[int(y_pred)],"**")
        pourcentage_prediction =1
        if y_pred == 1:
            pourcentage_prediction = int(predictions_Xray*100)
        elif y_pred == 0:
            pourcentage_prediction = (1 - int(predictions_Xray))* 100
        else:
            pourcentage_prediction = 0
        st.write("Le pourcentage de certitude est de : ", pourcentage_prediction)

else:
    st.title('Détails du modèle de détection de pneumonie')
    st.write(" ")
    st.write(" ")
    result()







