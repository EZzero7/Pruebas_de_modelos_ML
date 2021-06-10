import pandas as pd
import numpy as np
from sklearn import utils

from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from utils import Utils

class Models:
    #creamos un diccionario con la funcion que implementaremos
    def __init__(self):
        self.reg = {
            'SVR' : SVR(),
            'GRADIENT' : GradientBoostingRegressor()
        }

        #el segundo diccionario tendra los parametros para las funciones del primer diccionario
        self.params = {
            'SVR' : {
                'kernel' : ['linear', 'poly', 'rbf'],
                'gamma' : ['auto', 'scale'],
                'C' : [1.0,5.0,10.0]
            },'GRADIENT' : {
                'loss' : ['ls', 'lad'],
                'learning_rate' : [0.01,0.05,0.1]
            }
        }
    
    #ralizamos la funcion para elegir el mejor modelo con el dataset y los target ya preparados
    def grid_training(self,X,y):

        #creamos ambas variables para que guarden los resultados buscados
        best_score = 999
        best_model = None

        #realizamos una iteracion para probar con cada parametro en las funciones del diccionario reg
        for name, reg in self.reg.items():
            #gridsearchCV nos entrega los parametros mas precisos y los entrenamos con fit
            grid_reg = GridSearchCV(reg, self.params[name], cv=3).fit(X, y.values.ravel())
            score = np.abs(grid_reg.best_score_)
            #comparamos y elegimos el mejor modelo dado el score que nos dio
            if score < best_score:
                best_score = score
                best_model = grid_reg.best_estimator_

        #exportamos el mejor modelo para el proyecto con su respectivo score
        utils = Utils()
        utils.model_export(best_model, best_score)