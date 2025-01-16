import numpy as np
import pandas as pd
import collections #used for counting items of a list
from tensorflow import keras
from keras import layers
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
from tensorflow import keras
import json

"""

Genauigkeit des Modells:
Test accuracy: 0.8916000127792358

"""




"""
## Prepare the data
"""

# Model / data parameters
fashion_classes = 10
input_shape = (28, 28, 1)

"""
 the data, split between train and test sets
x_train: images for training
y_train: labels for training
x_test: images for testing the model
y_test: labels for testing the model
"""
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

"""
#download the images
"""
path = "C:/Users/Raffl Manuel/OneDrive - HTL Anichstrasse/HTL/5AHWII/INFI/Python/INFI_Abgaben_5AHWII/Aufgabe5/"

print ("storing images.....")
for i in range(1,20):

    plt.imshow(x_train[i])
    #plt.show()
    plt.savefig(path + str(i) +   " .png")



"""
# Scale images to the [0, 1] range
# Cast to float values before to make sure result ist float
"""
x_train = x_train.astype("float32") / 255
print(x_train.shape, "train samples")
x_test = x_test.astype("float32") / 255

# Make sure images have shape (28, 28, 1)
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

print(x_train.shape, "x_train shape:")
print(x_train.shape[0], "number of train samples")
print(x_test.shape[0], "number of test samples")

nr_labels_y = collections.Counter(y_train) #count the number of labels
print(nr_labels_y, "Number of labels")

# convert class vectors (the labels) to binary class matrices
y_train = keras.utils.to_categorical(y_train, fashion_classes)

y_labels = y_test #use this to leave the labels untouched
y_test = keras.utils.to_categorical(y_test, fashion_classes)

"""
## Build the model
"""

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

model = keras.Sequential(
    [
        keras.Input(shape=(784,)),
        layers.Dense(512, activation="relu"),
        layers.Dense(256, activation="relu"),
        layers.Dense(128, activation="relu"),
        layers.Dense(fashion_classes, activation="softmax"),
    ]
)

model.summary()

"""
## Train the model
"""

batch_size = 64
epochs = 16

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)

#draw the learn function
pd.DataFrame(history.history).plot(figsize=(8,5)) 
plt.show()
"""
## Evaluate the trained model
"""

score = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

"""
## Do some Preodictions on the test dataset and compare the results
"""

pred = model.predict(x_test)

print(pred[3]) #Prediction for image 1
pred_1 = np.argmax(pred[3])

# hier Bild anzeigen
plt.imshow(x_test[3].reshape(28,28))

print(pred_1)

for i in range(0,100):
    pred_i = np.argmax(pred[i]) # get the position of the highest value within the list
    print (y_labels[i], pred_i)


"""
How to load and save the model
"""

model.save("C:/Users/Raffl Manuel/OneDrive - HTL Anichstrasse/HTL/5AHWII/INFI/Python/INFI_Abgaben_5AHWII/Aufgabe5/model.h5")
model.save_weights("C:/Users/Raffl Manuel/OneDrive - HTL Anichstrasse/HTL/5AHWII/INFI/Python/INFI_Abgaben_5AHWII/Aufgabe5/model.weights.h5")

weights = model.get_weights()
j =json.dumps(pd.Series(weights).to_json(orient='values'), indent=3)
print(j)

model = keras.models.load_model("C:/Users/Raffl Manuel/OneDrive - HTL Anichstrasse/HTL/5AHWII/INFI/Python/INFI_Abgaben_5AHWII/Aufgabe5/model.h5")
model.load_weights("C:/Users/Raffl Manuel/OneDrive - HTL Anichstrasse/HTL/5AHWII/INFI/Python/INFI_Abgaben_5AHWII/Aufgabe5/model.weights.h5")

model_json = model.to_json()
print (model_json)