"""
:author Madmegsox1
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # stops tf beeing annoying 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import scipy
from PIL import Image

train = ImageDataGenerator(rescale=1 / 255)
valid = ImageDataGenerator(rescale=1 / 255)
class_names = ['cat', 'dog']  # sets class names
 # ⬇️ this creats the data sets for the training and validation
"""
train_ds = train.flow_from_directory("dog_cat\\training_set\\training_set\\",
                                     target_size=(200, 200),
                                     batch_size=32,
                                     class_mode='binary')

valid_ds = train.flow_from_directory("dog_cat\\test_set\\test_set\\",
                                     target_size=(200, 200),
                                     batch_size=32,
                                     class_mode='binary')
"""
# just uncomment this if you want to use it
# ⬇️ this is the model i used for the classification
""" 
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation="relu", input_shape=(200, 200, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(2)
])
"""

model = tf.keras.models.load_model('dog_cat/saved_model/cat_and_dog_model')  # loads the saved model 

model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # compiles the model
              optimizer="adam",
              metrics=['accuracy'])
"""  # trains the model, but isnt needed as the model is saved
epochs = 10 q
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=epochs
)
"""
model.save('dog_cat/saved_model/cat_and_dog_model') # saves the model

dir_path = "dog_cat/working_set"

for f in os.listdir(dir_path):  # loops throge all the images in "dog_cat/working_set"
    img = image.load_img(dir_path + "//" + f, target_size=(200, 200))  # resizes the images to the 200x200 format it was trained with

    # Image.open(dir_path+"//"+f).show()

    x = image.img_to_array(img)  # coverts the images to an array
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])  # does math

    pre = model(images, training=False)  # classifies the images
    for i, logits in enumerate(pre):
        class_id = tf.argmax(logits).numpy()  # converts it to a readble output
        p = tf.nn.softmax(logits)[class_id]
        name = class_names[class_id]
        print("{} is a {} ({:4.1f}%)".format(f, name, 100 * p))  # prints the output 
