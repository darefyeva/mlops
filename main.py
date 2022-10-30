from flask import Flask
from flask_restx import Api, Resource, fields, reqparse
import json
from models import prepare_data, fitting
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

app = Flask(__name__)
api = Api(app)

# словарь обученных моделей
trained_models = {13: 'test'}

models_classes = {'RandomForestRegressor': RandomForestRegressor(),
          'LinearRegression': LinearRegression()}

# подстановочные поля для обучения модели
# params_to_fit_model = api.model('Параметры для обучения модели',
#                      {'id_model': fields.Integer(description='Уникальный ID модели', example=13),
#                       'name_model': fields.String(description='Название модели для обучения', example='RandomForestRegressor'),
#                       'model_params': fields.Arbitrary(description='Параметры для обучения модели', example=json.dumps({'n_estimators': 100}))
#                         })

params_to_fit_model = reqparse.RequestParser()
params_to_fit_model.add_argument('id_model')
params_to_fit_model.add_argument('name_model')
params_to_fit_model.add_argument('model_params', help='(необязательное поле)')

@api.route('/all_available_models', methods=['GET'], doc={'description': 'Получить названия доступных моделей для обучения'})
class All_Available_Models(Resource):
    @api.response(200, 'OK')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        return 'Доступные модели регрессии для обучения: RandomForestRegressor, LinearRegression', 200


@api.route('/all_trained_models', methods=['GET'], doc={'description': 'Получить информацию об имеющихся обученных моделях'})
class All_Trained_Models(Resource):
    @api.response(200, 'OK')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def get(self):
        if trained_models == {}:
            return 'Нет доступных обученных моделей', 200
        else:
            return trained_models, 200

# обучает модель и сохраняет ее
@api.route('/fit_model', methods=['POST'], doc={'description': 'Обучить модель с выбранными параметрами'})
class Fit_Model(Resource):
    @api.expect(params_to_fit_model)
    @api.response(200, 'OK')
    @api.response(400, 'BAD REQUEST')
    @api.response(500, 'INTERNAL SERVER ERROR')
    def post(self):
        # id_model = api.payload['id_model']
        # name_model = api.payload['name_model']
        # model_params = api.payload['model_params']

        args = params_to_fit_model.parse_args()
        model_params = json.loads(args.model_params.replace("'", "\""))
        return model_params
        # if id_model in trained_models:
        #     return 'Модель с таким ID уже существует, выберите другое ID', 400
        # else:
        #     if name_model not in models_classes:
        #         return 'Такой модели для обучения не существует, попробуй RandomForestRegressor или LinearRegression', 400
        #     else:
        #         x, y = prepare_data('car_price_prediction')
        #         fitting(x, y, name_model, id_model, model_params)
        #         trained_models[id_model] = {'name_model': name_model, 'model_params': model_params}
        #         return 'Модель успешно обучена', 200


if __name__ == '__main__':
    app.run(debug=True)


