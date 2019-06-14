import numpy as np
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.applications.resnet50 import preprocess_input
from tensorflow.python.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array

num_classes = 2
resnet_weights_path = './data/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

train_dir = 'data/train'
val_dir = 'data/val'

my_new_model = Sequential()
my_new_model.add(ResNet50(include_top=False, pooling='avg',
                          weights=resnet_weights_path))
my_new_model.add(Dense(num_classes, activation='softmax'))

# Say not to train first layer (ResNet) model. It is already trained
my_new_model.layers[0].trainable = False

my_new_model.compile(
    optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


image_size = 224
data_generator = ImageDataGenerator(
    preprocessing_function=preprocess_input, horizontal_flip=True)


train_generator = data_generator.flow_from_directory(
    train_dir,
    target_size=(image_size, image_size),
    batch_size=20,
    class_mode='categorical')

validation_generator = data_generator.flow_from_directory(
    val_dir,
    target_size=(image_size, image_size),
    class_mode='categorical')

my_new_model.fit_generator(
    train_generator,
    epochs=2,
    steps_per_epoch=70,
    validation_data=validation_generator,
    validation_steps=1)
