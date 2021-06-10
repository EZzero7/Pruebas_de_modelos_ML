from utils import Utils
from models import Models



if __name__ == "__main__":
    #constructores
    utils = Utils()
    models = Models()

    #cargamos el dataset
    data = utils.load_from_csv('./in/felicidad.csv')
    #preparamos el dataset
    X, y = utils.features_target(data, ['score','rank','country'],['score'])
    # entrenamos el dataset
    models.grid_training(X,y)

