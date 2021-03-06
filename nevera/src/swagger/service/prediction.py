from flask import request
from flask_restplus import Namespace, Resource, fields

namespace = Namespace('funciono', description = 'Image classifier')
"""
metrics_model = namespace.model('metrics', {
    'accuracy': fields.Float(
        readonly=True,
        description='Model accuracy'
    ),
    'execution_time': fields.Integer(
        readonly=True,
        description='execution time'
    ),
})
"""
metrics_model = namespace.model('metrics', {'metrics':"hola, he medio, jaja, si, medio"})

prediction_model = namespace.model('prediction', {
    'creationDate': fields.String(
        readonly=True,
        description='Timestamp'
    ),
    'message': fields.String(
        readonly=True,
        description='base64 image'
    )
})


"""
get_prediction_model = namespace.model('get_prediction', {
    'id': fields.String(
        required=True,
        description='get prediction by ID'
    )
})
"""
get_prediction_model = namespace.model('get_prediction', {"get_prediction_model":"hola, he io por las predisione"})

@namespace.route('')
class prediction(Resource):

    @namespace.marshal_list_with(prediction_model)
    @namespace.response(500, 'Internal Server error')
    def post(self):
        '''Prediction endpoint'''
        return prediction_model
    @namespace.marshal_list_with(get_prediction_model)
    def get(self):
        '''Prediction endpoint'''
        return get_prediction_model
