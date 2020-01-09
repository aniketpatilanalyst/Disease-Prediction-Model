from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")


# Initialising the CNN
classifier = Sequential()

# Step1 - Convolution
# Input Layer/dimensions
# Step-1 Convolution
# 64 is number of output filters in the convolution
# 3,3 is filter matrix that will multiply to input_shape=(64,64,3)
# 64,64 is image size we provide
# 3 is rgb
classifier.add(Convolution2D(64,3,3, input_shape=(64,64,3), activation='relu'))

# Step2 - Pooling
#Processing
# Hidden Layer 1
# 2,2 matrix rotates, tilts, etc to all the images
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding a second convolution layer
# Hidden Layer 2
# relu turns negative images to 0
classifier.add(Convolution2D(64,3,3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# step3 - Flattening
# converts the matrix in a singe array
classifier.add(Flatten())

# Step4 - Full COnnection
# 128 is the final layer of outputs & from that 1 will be considered ie dog or cat
classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))
# sigmoid helps in 0 1 classification

# Compiling the CNN
classifier.compile(optimizer ='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Deffining the Training and Testing Datasets
from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        '.../dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        '.../dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# nb_epochs how much times you want to back propogate
# steps_per_epoch it will transfer that many images at 1 time
# & epochs means 'steps_per_epoch' will repeat that many times
classifier.fit_generator(
        training_set,
        steps_per_epoch=5000,
        nb_epoch=10,
        validation_data=test_set,
        nb_val_samples=1000)

import numpy as np
from PIL import image
# Verifing ouor Model by giving samples of cell to detect malaria
test_image = image.load_img('/kaggle/input/files1/Malaria Cells/single_prediction/Parasitised.png', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'Uninfected'
else:
    prediction = 'Parasitised'
prediction
