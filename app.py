from flask import Flask, render_template, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from skimage.transform import resize
from flask_cors import CORS  # Add this line to handle cross-origin resource sharing
from matplotlib import image

import os, shutil
import numpy as np

from matplotlib import image, pyplot
from skimage.transform import resize

import tensorflow as tf
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D,MaxPool2D,Dense,Flatten,Dropout,MaxPooling2D,Activation

from keras import callbacks
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score, precision_score, f1_score, recall_score,classification_report,roc_curve, auc
from sklearn.utils import class_weight

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
import matplotlib.pyplot as plt
from keras.optimizers import Adam
from keras.callbacks import LearningRateScheduler
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, Reshape, GRU

app = Flask(__name__)
CORS(app)  # Allow cross-origin resource sharing

# Load the model
def get_complex_model_with_gru():
    model = Sequential([
        Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(256, (3, 3)),
        Activation('relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Activation('relu'),
        Dropout(0.5),
        Dense(128, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        Activation('relu'),
        Dropout(0.5),
        Reshape((1, 128)),  # Reshape to add the time dimension
        GRU(units=64, return_sequences=True),
        GRU(units=64),
        Dense(1, activation='sigmoid')
    ])

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy']
    )
    return model

# @app.before_first_request
def load_model():
    global model
    model = get_complex_model_with_gru()
    model.load_weights("checkpoint-0061.hdf5")
load_model()

def preprocess_image(image_path):
    # Load and preprocess the image
    img = image.imread(image_path)
    img_resized = resize(img, (224, 224, 3))
    img_resized = np.expand_dims(img_resized, axis=0)
    return img_resized

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the image from the request
    image_file = request.files['image']
    
    # Save the uploaded image
    image_path = 'uploaded/image.jpg'  # Update with your desired path
    image_file.save(image_path)

    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Make predictions
    prediction_prob = model.predict(processed_image)
    prediction = np.round(prediction_prob)

    # Return the result
    result = {'prediction': int(prediction[0][0]), 'probability': float(prediction_prob[0][0])}
    return jsonify(result)

# if __name__ == '__main__':
#     app.run(debug=True)
