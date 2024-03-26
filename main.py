# Importing necessary libraries for building and training the model

# Importing layers and model from Keras for building the neural network
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

# Path to the directory containing the training set
train_set = "D:/projects/fruit/train/train"

# Importing ImageDataGenerator from Keras for data augmentation
from keras.preprocessing.image import ImageDataGenerator

# Creating an ImageDataGenerator object for augmenting the training data
train_datagen = ImageDataGenerator(
    rescale = 1./255,             # Rescale pixel values to [0,1]
    rotation_range = 45,          # Range for random rotations
    horizontal_flip = True,       # Randomly flip images horizontally
    shear_range = 0.3,            # Shear angle in counter-clockwise direction
    validation_split = 0.2,       # Fraction of training data to use for validation
    zoom_range = 0.3              # Range for random zoom
)

# Generating training and validation data using flow_from_directory method
train_data = train_datagen.flow_from_directory(
    train_set,
    target_size = (100, 100),     # Resize images to 100x100 pixels
    class_mode = 'categorical',   # Mode for classifying labels
    batch_size = 64,              # Number of samples per batch
    subset = "training"           # Specify training subset
)

valid_data = train_datagen.flow_from_directory(
    train_set,
    target_size = (100, 100),     # Resize images to 100x100 pixels
    class_mode = 'categorical',   # Mode for classifying labels
    batch_size = 64,              # Number of samples per batch
    subset = "validation"         # Specify validation subset
)

# Importing VGG16 model architecture from Keras applications
from keras.applications.vgg16 import VGG16

# Loading the VGG16 model with pre-trained weights
vgg16 = VGG16(input_shape = (100,100, 3), weights = 'imagenet', include_top = False)

# Freezing the existing weights in the VGG16 model
for layer in vgg16.layers:
    layer.trainable = False

# Adding fully connected layers on top of the VGG16 base
flatten = Flatten()(vgg16.output)
dense = Dense(256, activation = 'relu')(flatten)
dense = Dropout(0.5)(dense)
dense = Dense(100, activation = 'relu')(dense)
dense = Dropout(0.3)(dense)

# Output layer for classifying fruit categories
prediction = Dense(33, activation='softmax')(dense)

# Creating the final model with VGG16 as base and added layers
model = Model(inputs=vgg16.input, outputs=prediction)

# Displaying summary of the model architecture
model.summary()

# Compiling the model with categorical crossentropy loss and Adam optimizer
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Training the model using the augmented training data and validating on validation data
history = model.fit_generator(train_data, validation_data=valid_data, epochs=15)

# Saving the trained model
model.save("fruit.h5")
