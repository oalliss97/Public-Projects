import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import ImageFile
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Activation, Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.applications import ResNet50

############# Preprocessing and Loading #################################################################################################################
ImageFile.LOAD_TRUNCATED_IMAGES = True
im_dir = Path(os.path.join(os.getcwd(), "Chocolate Lab Model/Images"))

batch_size = 256
epochs = 1000
IMG_HEIGHT = 224
IMG_WIDTH = 224
image_generator = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_data_gen = image_generator.flow_from_directory(batch_size=batch_size, directory=im_dir, shuffle=True, target_size=(
    IMG_HEIGHT, IMG_WIDTH), class_mode='binary', subset='training')
test_data_gen = image_generator.flow_from_directory(batch_size=batch_size, directory=im_dir, shuffle=True, target_size=(
    IMG_HEIGHT, IMG_WIDTH), class_mode='binary', subset='validation')

print("Class Indices: ", test_data_gen.class_indices)

############### Does User want to see images? ##############################################################################
sample_images, _ = next(train_data_gen)

def plotImages():
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(sample_images[:10], axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


print("Do you want to preview the resized images? (Yes/No)")
answer = input()
if answer == "Yes":
    plotImages()
elif answer == "No":
    print("Okay, building model now!")
else:
    print("Invalid response")

############# Model Building and Fitting #################################################################################################################
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
resnet = ResNet50(include_top=True)

model = Sequential()
model.add(resnet)
#model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(32, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Conv2D(64, (3, 3)))
#model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
#model.add(Dense(64))
#model.add(Activation('relu'))
#model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.layers[0].trainable = False
#model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(
    train_data_gen,
    epochs=epochs,
    validation_data=test_data_gen,
    callbacks=[es])

model.save(Path(os.path.join(os.getcwd(), "Chocolate Lab Model/chocolate_lab.h5")), include_optimizer = True)

############### Does User want to see performance? ##############################################################################
print("Do you want to see the Loss and Accuracy? (Yes/No)")
answer = input()
if answer == "Yes":
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(epochs)
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Testing Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Testing Accuracy')
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Testing Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Testing Loss')
    plt.show()
elif answer == "No":
    print("Okay, we are now finished!")
else:
    print("Invalid response")
