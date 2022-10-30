import pandas as pd
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

def prepare_data(file_name: str):
    """
    Подготовка данных для обучения и предсказания.
    Данные представляют собой датафрейм с признаками машин.

    Param:
    file_name: str

    Return:
    x: pd.DataFrame
    y: pd.Series

    """
    df = pd.read_csv(file_name, sep=';')
    y = df['Price']
    x = df.drop(['Price'], axis=1)[['Mileage', 'Cylinders', 'Airbags', 'Prod year']]
    return x, y


def fitting(x_train: pd.DataFrame, y_train: pd.Series, model_name: str, id_model: int, model_params: dict):
    """
    На вход принимает признаки и таргет, название модели, которую нужно обучить и параменты этой модели.
    Обучение модели на train.
    Возвращает обученную модель.

    Param:
    X_train: pd.DataFrame
    y_train: pd.Series
    model_name :str
    model_params :dict

    Return:
    model

    """
    models = {'RandomForestRegressor': RandomForestRegressor(), 'LinearRegression': LinearRegression()}
    for param in model_params:
        if param not in models[model_name].get_params().keys():
            return f'Параметр {param} у модели {model_name} не найден',  400

    model = models[model_name].set_params(**model_params)
    model.fit(x_train, y_train)
    with open('../fitted_models/' + str(id_model), 'wb') as file:
        pickle.dump(model, file)


# def prediction(fitted_model, X_test: pd.DataFrame):
# """
# """
#     y_pred = fitted_model.predict(X_test)
#
#     return y_pred