import pandas as pd
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
import base64
from datetime import datetime
from skimage import color
from skimage import io
import requests
import os


import tensorflow as tf
#import tf.keras.models.load_model
from keras.models import model_from_json
import tensorflow.keras.applications.inception_resnet_v2
import tensorflow.python.keras.applications.inception_resnet_v2 
import numpy as np
from numpy import expand_dims
from PIL import Image
from keras.layers import Lambda
import PIL
from tensorflow.keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input as preprocess_vgg16

from swagger.app import blueprint as app_endpoints
from datetime import datetime

### Comment these lines if swagger is not working. 
# app
flask_app = Flask(__name__)
api = Api(app = flask_app)

#app.config["RESTPLUS_MASK_SWAGGER"] = False
#app.register_blueprint(app_endpoints)

#name_space = app.namespace('funciono', description = 'Main API')


#Load models
from keras.models import load_model
model = load_model('./models/clasificador_nevera.h5')


 # Assigning label names to the corresponding indexes
class_dict = ['Actimel','Alpro','coca-cola','Florette','Leche','Schweppes','Tomate']


class prediction_test(Resource):
    def get(self):
        return{"queloque": "tu sabras, amigo, alla tu, esto es un get"}
#en este def debería de entrar la imagen como argumento
    def post(self):
        data = request.get_json(force = True)
        img  = data["message"]
        base64_img_bytes = img.encode("utf-8")
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        with open("imagenes/prueba.jpg", "wb") as file_to_save:  # bucket_
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)
        # procesado total-----
        img= PIL.Image.open('imagenes/prueba.jpg')

        img = img.resize((224, 224))
        data = expand_dims(image.img_to_array(img), 0)
        data = preprocess_vgg16(data)
        preds = model.predict(data)
        pred = np.argmax(preds)
        pred = class_dict[pred]
    
    
        return jsonify({"about": "Prueba guardada e imagen predicha, jaja que bien, la verdad. Vaya alivio. Estoy a punto de llorar","predicción":pred})


#{'carpeta':'Train','producto':'Actimel','imagen':'jksfb'}
class almacenar(Resource):
    def get(self):
        return{"queloque": "tu sabras, amigo, alla tu, esto es un get"}
    def post(self):
        data = request.get_json(force = True)
        carpeta = data['carpeta']
        producto = data['producto']
        img  = data["imagen"]

        dt = datetime.now()
        ts = datetime.timestamp(dt)
        name = round(ts) 
        base64_img_bytes = img.encode("utf-8")
        decoded_image_data = base64.decodebytes(base64_img_bytes)
        with open(f'train_images/{carpeta}/{producto}/{name}.jpg', "wb") as file_to_save:  
            decoded_image_data = base64.decodebytes(base64_img_bytes)
            file_to_save.write(decoded_image_data)
        return jsonify({"about":"Completado con exito", "carpeta":carpeta,"producto":producto})
class accuracy(Resource):
    def get(self):
        return{"queloque": "tu sabras, amigo, alla tu, esto es un get"}
    def post(self):
        os.system('python3 validation.py')
        return jsonify({"about":"Completado con exito"})
class retrain(Resource):
    def get(self):
        return{"queloque": "tu sabras, amigo, alla tu, esto es un get"}
    def post(self):
        os.system('python3 entrene.py')
        return jsonify({"about":"Completado con exito"})


api.add_resource(prediction_test, "/funciono")
api.add_resource(almacenar, "/almacenar")
api.add_resource(accuracy, "/accuracy")
api.add_resource(retrain, "/retrain")

if __name__ == "__main__":
    flask_app.run(port=5000, debug=True, host="0.0.0.0")