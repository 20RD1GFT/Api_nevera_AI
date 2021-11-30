from tensorflow.keras.applications.vgg16 import VGG16
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input as preprocess_vgg16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
from keras.layers import Dense,GlobalAveragePooling2D,Flatten
#from keras.applications import VGG16
import glob
import os.path
from PIL import Image

class run_training():
    #Data load
    train_dir = 'train_images/train'
    validation_dir = 'train_images/val'
    # Data generators
    train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.2,zoom_range=0.2,horizontal_flip=True,fill_mode='nearest')

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(train_dir,target_size=(224, 224),batch_size=16,class_mode='categorical',subset ='training')

    validation_generator = test_datagen.flow_from_directory(validation_dir,target_size=(224, 224),batch_size=16,class_mode='categorical',subset = 'validation')
    #Train config
    #we define earlystopping and a modelcheckpoint
    earlystopping = EarlyStopping(monitor = 'loss', patience = 20)

    # save the best model with lower loss
    checkpointer = ModelCheckpoint(filepath = "./models/weights.hdf5", save_best_only = True)

    step_size_train=(train_generator.n//train_generator.batch_size)


    #Base model load
    base_model=VGG16(weights='imagenet',include_top=False)
    num_classes = 7
    #We define the layers that are to be trained for the module
    x=base_model.output
    x=GlobalAveragePooling2D()(x)
    x=Dense(512,activation='relu')(x)
    preds=Dense(num_classes,activation='softmax')(x) #we keep the softmax for the purpose
    from keras.models import Model

    model=Model(inputs=base_model.input,outputs=preds)
    #Model compiling and fitting
    model.compile(optimizer='Adam',loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
    steps_per_epoch=step_size_train,
    validation_data=validation_generator,
    validation_steps=validation_generator.n//validation_generator.batch_size,
    epochs=1,
    callbacks = [checkpointer, earlystopping])
    #Save model
    #we need to save the model with a timestamp so we can retrieve the latest model for later.

    dt = datetime.now()
    ts = datetime.timestamp(dt)
    name = round(ts)
    model_name = (f'clasificador_nevera{name}.h5')
    model.save(f'./models/{model_name}')
    
if __name__ == "__main__":
    run_training()