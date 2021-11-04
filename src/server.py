import pandas as pd
from flask import Flask, jsonify, request
from flask_restplus import Api, Resource
import base64
from datetime import datetime
from skimage import color
from skimage import io
import requests


import tensorflow as tf
from keras.models import model_from_json
import tensorflow.keras.applications.inception_resnet_v2
import tensorflow.python.keras.applications.inception_resnet_v2 
import numpy as np
from PIL import Image
from keras.layers import Lambda
import PIL

from swagger.app import blueprint as app_endpoints

### Comment these lines if swagger is not working. 
# app
flask_app = Flask(__name__)
api = Api(app = flask_app)

#app.config["RESTPLUS_MASK_SWAGGER"] = False
#app.register_blueprint(app_endpoints)

#name_space = app.namespace('funciono', description = 'Main API')

from keras.models import load_model
model = load_model("./models/modelo2.h5")


 # Assigning label names to the corresponding indexes
labels = {
    0: 'Bread', 
    1: 'Dairy product', 
    2: 'Dessert', 
    3: 'Egg', 
    4: 'Fried food', 
    5: 'Meat',
    6: 'Noodles-Pasta',
    7: 'Rice', 
    8: 'Seafood',
    9: 'Soup',
    10: 'Vegetable-Fruit'
}


class prediction_test(Resource):
    def get(self):
        return{"queloque": "tu sabras, amigo, alla tu, esto es un post"}
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
        #resizing the image to (256,256)
        img = img.resize((256,256))
        #converting image to array
        img = np.asarray(img, dtype= np.float32)
        #normalizing the image
        img = img / 255
        #reshaping the image in to a 4D array
        img = img.reshape(-1,256,256,3)
        #making prediction of the model
        predict = model.predict(img)
        #getting the index corresponding to the highest value in the prediction
        predict = np.argmax(predict)
        respond = labels[predict]
        return jsonify({"about": "Prueba guardada e imagen predicha, jaja que bien, la verdad. Vaya alivio. Estoy a punto de llorar",
                        'respuesta': respond})

api.add_resource(prediction_test, "/funciono")

if __name__ == "__main__":
    flask_app.run(port=5000, debug=True, host="0.0.0.0")