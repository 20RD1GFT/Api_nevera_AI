import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.mobilenet import preprocess_input as preprocess_vgg16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from datetime import datetime
import glob
import os.path


def run_training():


#Model load
#we need to scan the directory and load the latest model
#This snippet of code allows us to scan the folder for the latest model. This should be later addressed since we don't have
#any model deleting control measurements


    folder_path = r'.\models'
    file_type = '\*h5'
    files = glob.glob(folder_path + file_type)
    most_recent_model = max(files, key=os.path.getctime)
    model = keras.models.load_model(most_recent_model)

    #Data load
    train_generator=train_datagen.flow_from_directory('./train_images/train',
                                                      target_size=(224,224),
                                                      #default parameters
                                                      color_mode='rgb',
                                                      batch_size=16,
                                                     class_mode='categorical',
                                                      subset='training',
                                                      shuffle=True)
    #Incluimos un set de validacion.
    validation_generator=train_datagen.flow_from_directory('./train_images/train',
                                                      target_size=(224,224),
                                                      color_mode='rgb',
                                                      batch_size=16,
                                                     class_mode='categorical',
                                                      shuffle=True,
                                                     subset='validation')

    #Train config
    #we define earlystopping and a modelcheckpoint
    earlystopping = EarlyStopping(
        monitor = 'loss',
        patience = 20
    )

    # save the best model with lower loss
    checkpointer = ModelCheckpoint(
        filepath = "./models/weights.hdf5",
        save_best_only = True
    )

    #Model compiling and fitting
    model.compile(optimizer='Adam',loss='categorical_crossentropy',
                  metrics=['accuracy'])

    step_size_train=train_generator.n//train_generator.batch_size
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train,
                        validation_data=validation_generator,
                        validation_steps=validation_generator.n//validation_generator.batch_size,
                        epochs=6,
                        callbacks = [checkpointer, earlystopping])

    #Save model
    #we need to save the model with a timestamp so we can retrieve the latest model for later.

    dt = datetime.now()
    ts = datetime.timestamp(dt)
    name = round(ts)

    model_name = ("clasificador_nevera" + name + ".h5")
    model.save("./models/" + model_name)
    
if __name__ == "__main__":
    run_training()