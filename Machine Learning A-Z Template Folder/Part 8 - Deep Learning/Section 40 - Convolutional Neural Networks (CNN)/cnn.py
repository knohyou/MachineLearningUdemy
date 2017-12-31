# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
# Fast for numerical computation
# Run on GPU and CPU. More powerful more floating points. Use with numpy. Run parallel 
# Simple neural network can just use CPU

# Installing Tensorflow
# Install Tensorflow from the website: https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html
# Run both GPU and CPU
 
# Both Theano and tensorflow used for research purposes

# Installing Keras
# pip install --upgrade keras
# Keras a library based on Theano and Tensorflow
# Efficiently create ANN

from timeit import default_timer as timer
start = timer()

# Part 1 - Building the Convolution Neural Network (CNN)
from keras.models import Sequential # Initialize neural network Sequence of networks
from keras.layers import Convolution2D # First step in CNN convolutional layer
from keras.layers import MaxPooling2D # Step 2 is the pooling step
from keras.layers import Flatten
from keras.layers import Dense # for fully processed

# Initilize the CNN
classifier = Sequential() # First step to setup initalize layer

# Step 1 - Convolutional layer
classifier.add(Convolution2D(filters = 32,
                             kernel_size = (3, 3),
                             input_shape = (64,64,3),
                             activation = 'relu'))


# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2,2)))

# Add a second convolutional layer
classifier.add(Convolution2D(filters = 32,
                             kernel_size = (3, 3),
                             activation = 'relu')) # Dont need to include the input shape kares will know
classifier.add(MaxPooling2D(pool_size = (2,2)))
# Adding more feature detector would help. Filters = 64

# Add a third convolutional layer
classifier.add(Convolution2D(filters = 32,
                             kernel_size = (3, 3),
                             activation = 'relu')) # Dont need to include the input shape kares will know
classifier.add(MaxPooling2D(pool_size = (2,2)))
# Adding more feature detector would help. Filters = 64

# Step 3 - Flattening 
classifier.add(Flatten())

# Step 4 - Create ANN
classifier.add(Dense(units = 128,
                     activation = 'relu')) # number of hidden node 
classifier.add(Dense(units = 1,
                     activation = 'sigmoid')) # number of hidden node  Output layer as 1 

# Compiling the CNN
classifier.compile(optimizer = 'adam', # Stochastic gradient descent
                   loss = 'binary_crossentropy',
                   metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
# Perform image agumentation 
# Prevent overfitting on training set
# Perform random transformation on the current images to add more dataset for training even with small dataset 

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')
# Even better would be increasing the target size to 128 and 128


classifier.fit_generator(training_set,
                        steps_per_epoch=8000,
                        epochs=25,
                        validation_data=test_set,
                        validation_steps=2000)


# elapsed time in minutes
end = timer()
print("Elapsed time in minutes")
print(0.1*round((end - start)/6))
# end of work message
import os
os.system('say "your program has finished"')
