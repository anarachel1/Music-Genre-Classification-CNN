import json
import pickle
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_data(dataset_path):
    with open (dataset_path, "r") as fp:
        data = json.load(fp)
        
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    # input.shape = 0-segment, 1- interval, 3-coefficient magnitute
    
    return x, y

    
def buil_model (input_shape):
    model= keras.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(512, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(11, activation="softmax")
        # 10 = 10 possibles targets
    ])
    return model


def plot_history(history):
    fig, ax = plt.subplots(2)
    ax[0].plot(history.history["accuracy"], label = "train accuracy")
    ax[0].plot(history.history["val_accuracy"], label = "test accuracy")
    ax[0].set_ylabel("Accuracy")
    ax[0].set_title("Accuracy eval")
    ax[0].legend(loc="lower right")

    ax[1].plot(history.history["loss"], label = "train error")
    ax[1].plot(history.history["val_loss"], label = "test error")
    ax[1].set_ylabel("Error")
    ax[1].set_title("Error eval")
    ax[1].legend(loc="upper right")
    
    plt.show()
    
    
def RNN_train(data, epochs, batch):
    #import data
    x, y = load_data(data)
    
    # split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.3)
    
    # buil de model
    model = buil_model((x.shape[1], x.shape[2]))
    
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    # describe the model
    model.summary()
    
    # train network
    history = model.fit( x_train, y_train, 
                        validation_data=(x_test, y_test),
                        epochs = epochs,
                        batch_size = batch)
    plot_history(history)
    
    # save model
    with open("rnn_model.pkl", "wb") as file:
        pickle.dump(model, file)
    
    
def RNN_predict(x):
    with open("rnn_model.pkl", "rb") as file:
        model = pickle.load(file)
        
    pred = model.predict(x)
    print( f"The expected genrse is {pred}")
    
        
    
if __name__ == "__main__":
    
    data_path = "data.json"
        
    #RNN_train(data_path, epochs = 2, batch = 32)
    
    # making a inference, song n100
    with open (data_path, "r") as fp:
        data = json.load(fp)
        
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    test_song = x[100]
    
    RNN_predict(test_song)
    print(f"Expected label for the test song is {y[100]}")
    
    