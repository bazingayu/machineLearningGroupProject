import os

from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from sklearn.metrics import confusion_matrix
import numpy as np
import cv2
from matplotlib import pyplot as plt

num_classes = 2

def standardization(data):
    mu = np.mean(data)
    sigma = np.std(data)
    print(mu, sigma)
    return (data - mu) / sigma

def svm_classification(Xtrain, Xtest, ytrain, ytest):
    model = LinearSVC(max_iter=10000)
    model = model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtrain)
    print(confusion_matrix(ytrain, ypred))

def knn_classification(Xtrain, Xtest, ytrain, ytest):
    model = KNeighborsClassifier(n_neighbors=5, weights='uniform')
    model = model.fit(Xtrain, ytrain)
    ypred = model.predict(Xtest)
    print(confusion_matrix(ytest, ypred))

def compile_trainingset(positive_path, negative_path):
    train_x = []
    train_y = []
    for filename1 in os.listdir(positive_path):
        filename = os.path.join(positive_path, filename1)
        image = cv2.imread(filename)
        image = cv2.resize(image, (32, 32))
        train_x.append(image)
        train_y.append(1)
    for filename1 in os.listdir(negative_path):
        filename = os.path.join(negative_path, filename1)
        image = cv2.imread(filename)
        image = cv2.resize(image, (32, 32))
        train_x.append(image)
        train_y.append(0)
    train_x = np.array(train_x)
    train_y = np.array(train_y)
    return train_x, train_y



def CNN(Xtrain, Xtest, ytrain, ytest):
    plt.figure()
    model = keras.Sequential()
    model.add(Conv2D(8, (3, 3), padding='same', input_shape=Xtrain.shape[1:], activation='relu'))
    model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=2, padding='same'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax', kernel_regularizer=regularizers.l1(0.0001)))
    model.compile(loss="categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()

    batch_size = 128
    epochs = 100
    history = model.fit(Xtrain, ytrain, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    ypred = model.predict(Xtest)
    # print(confusion_matrix(ytest, ypred))
    model.save("cme.model")
    plt.subplot(211)
    plt.plot(history.history['accuracy'], label = f'train accuracy ')
    plt.plot(history.history['val_accuracy'], label = f'val accuracy ')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    # plt.legend(['train', 'val'], loc='upper left')
    plt.subplot(212)
    plt.plot(history.history['loss'], label = f'train loss ')
    plt.plot(history.history['val_loss'], label = f'val loss ')
    plt.title('model loss')
    plt.ylabel('loss'); plt.xlabel('epoch')
    plt.show()

















def classification_all_classifiers():
    # for lbp
    # lbp_features = np.load("hist_features.npy")
    #
    # lbp_X = lbp_features[:, :-1]
    # lbp_y = lbp_features[:, -1]
    # print(lbp_X[1], lbp_X[-1])
    # lbp_X = standardization(lbp_X)
    # print(sum(lbp_X[10]))
    # # Xtrain, Xtest, ytrain, ytest = train_test_split(lbp_X, lbp_y, test_size=0.1, random_state=0)
    # print("start svm")
    # svm_classification(lbp_X, lbp_X, lbp_y, lbp_y)
    # print("start knn")
    # knn_classification(lbp_X,  lbp_X, lbp_y, lbp_y)

    train_x, train_y = compile_trainingset("D:/BaiduNetdiskDownload/image/CMEImages/CME_polar_crop", "D:/BaiduNetdiskDownload/image/CMEImages/NoCME_polar_crop")
    x_train = train_x.astype("float32") / 255.0
    train_y = keras.utils.to_categorical(train_y, num_classes)
    CNN(x_train, x_train, train_y, train_y)

classification_all_classifiers()