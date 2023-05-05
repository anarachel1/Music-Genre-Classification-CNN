import json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATASET_PATH = "data.json"
# load data 
def load_data(dataset_path):
    with open (dataset_path, "r") as fp:
        data = json.load(fp)
        
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])
    # input.shape = 0-segment, 1- interval, 3-coefficient magnitute
    
    return inputs, targets



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
    
    
if __name__ == "__main__":
    
    #import data
    inputs, targets = load_data(DATASET_PATH)
    
    # split the data into train and test sets
    input_train, input_test, target_train, target_test = train_test_split(inputs,
                                                                          targets,
                                                                          test_size= 0.3)
    # buil de model
    model= keras.Sequential([
        keras.layers.Flatten(input_shape=(inputs.shape[1], inputs.shape[2])),
        keras.layers.Dense(512, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(256, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu", kernel_regularizer = keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(11, activation="softmax")
        # 10 = 10 possibles targets
    ])
    
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    model.summary()
    
    # train network
    history = model.fit( input_train, target_train, 
                        validation_data=(input_test, target_test),
                        epochs = 20,
                        batch_size = 32)
    plot_history(history)
    