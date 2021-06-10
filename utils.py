import pandas as pd
import joblib


class Utils:
    #Funcion para cargar datos provenientes de un archivo csv
    #Pasando como parametros la direccion del archivo
    def load_from_csv(self, path):
        return pd.read_csv(path)

    #Funcion para cargar datos provenientes de una base de datos mysql
    def load_from_mysql(self):
        pass
    
    #Funcion para dividir los datos de los targuets que vamos a utilizar
    #Pasando como parametros el dataset, la columnas que eliminaremos y la columna que utilizaremos de targuet
    def features_target(self, dataset, drop_cols, y):
        #X posee el dataset sin las columnas que queremos eliminar y sin la columna que utilizaremos de targuet
        X = dataset.drop(drop_cols, axis = 1)
        #y posee los datos de la columnas que ultilizamos el targuet
        y = dataset[y]
        return X,y

    #recibe el modelo y el score y exporta el modelo en un archivo pkl
    def model_export(self, clf, score):
        print(score)
        joblib.dump(clf, './models/best_model.pkl')