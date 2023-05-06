import json
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


DATA_PATH = "data.json"

# load data 
def load_data(dataset_path):
    with open (dataset_path, "r") as fp:
        data = json.load(fp)
        
    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    # input.shape = 0-segment, 1- interval, 3-coefficient magnitute
    
    return x, y

def preprare_data(test_size, validation_size):
    x,y = load_data(DATA_PATH)
    
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = test_size)
    
    x_train, x_validation, y_train, y_validation = train_test_split(x_train, y_train, test_size= validation_size)

    # cnn for audio, inputs 3D array -> (130 time bins, 13 mfcc coef, 1 channel)
    # the input needs 4D array -> number of samples (samples, time bins,  mfcc coef, channel)
    x_train = x_train[..., np.newaxis]
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]
    
    return  x_train, x_test, x_validation, y_train, y_test, y_validation
    
    
def build_model( input_shape):
    print("input: ", input_shape)
    model = keras.Sequential()
    # 1 convulutional layer
    model.add(keras.layers.Conv2D(32, (3,3), activation = "relu", input_shape = input_shape))
    model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    
    # 2 convulutional layer
    model.add(keras.layers.Conv2D(32, (3,3), activation = "relu"))
    model.add(keras.layers.MaxPool2D((3,3), strides = (2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    
    # 3 convulutional layer
    model.add(keras.layers.Conv2D(32, (2,2), activation = "relu"))
    model.add(keras.layers.MaxPool2D((2,2), strides = (2,2), padding = 'same'))
    model.add(keras.layers.BatchNormalization())
    
    # flatten and dense layer
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(64, activation = "relu"))
    model.add(keras.layers.Dropout(0.3))
    
    # output layer
    model.add(keras.layers.Dense(11, activation = 'softmax'))
    
    return model


def predict(model, x, y):
    x = x[np.newaxis, ...]
    prediction = model.predict(x)
    predicted_index = np.argmax(prediction, axis=1)
    print(f"Expected index {y} and the predicted index {predicted_index}")


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
    
    x_train, x_test, x_validation, y_train, y_test, y_validation = preprare_data(0.25 , 0.2)
    
    input_shape = (x_train.shape[1], x_train.shape[2], 1)
    model = build_model(input_shape)
    
    # compile network
    optimizer = keras.optimizers.Adam(learning_rate = 0.0001)
    model.compile(optimizer = optimizer,
                  loss = "sparse_categorical_crossentropy",
                  metrics = ["accuracy"])
    
    model.summary()
    
    # train network
    model_object = model.fit( x_train, y_train, validation_data=(x_validation, y_validation),
                        epochs = 30,
                        batch_size = 32)
    plot_history(model_object)
    
    #evaluate the cnn on the test set
    test_error, test_accuracy = model.evaluate( x_test, y_test, verbose=1)
    print(f"Accuracy on test set is : {test_accuracy}")
    
    # prediction 
    x = x_test[100]
    y = y_test[100]
    predict(model, x,y)