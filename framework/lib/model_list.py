import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoCV
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import LassoLars
from sklearn.linear_model import OrthogonalMatchingPursuit
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import PassiveAggressiveRegressor
from sklearn.linear_model import TheilSenRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ARDRegression

from sklearn import svm

from sklearn.neighbors import KNeighborsRegressor

ensemble_params = {
      'bootstrap': [True],
      'max_depth': [80, 90, 100, 110],
      'max_features': [2, 3],
      'min_samples_leaf': [3, 4, 5],
      'min_samples_split': [8, 10, 12],
      'n_estimators': [100, 200, 300, 1000]
      }

alphas = np.logspace(-4, -0.5, 30)
linear_model_params = [{'alpha': alphas}]

def get_models():
  models = [
    ( 'RandomForestRegressor',
      RandomForestRegressor(),
      ensemble_params
    ),
    ( 'ExtraTreesRegressor',
      ExtraTreesRegressor(),
      ensemble_params
    ),
    ( 'Lasso',
      Lasso(random_state=0, max_iter=10000),
      linear_model_params
    ),
    ( 'ElasticNet',
      ElasticNet(random_state=0, max_iter=10000),
      linear_model_params
    ),
    ( 'LassoLars',
      LassoLars(max_iter=10000),
      linear_model_params
    ),
    ( 'OrthogonalMatchingPursuit',
      OrthogonalMatchingPursuit(),
      {}
    ),
    ( 'BayesianRidge',
      BayesianRidge(),
      {}
    ),
    ( 'Ridge',
      Ridge(),
      {}
    ),
    ( 'SGDRegressor',
      SGDRegressor(random_state=0, max_iter=10000, tol=1e-3),
      linear_model_params
    ),
    ( 'PassiveAggressiveRegressor',
      PassiveAggressiveRegressor(random_state=0, max_iter=10000),
      {}
    ),
    ( 'TheilSenRegressor',
      TheilSenRegressor(random_state=0, max_iter=10000),
      {}
    ),
    ( 'LinearRegression',
      LinearRegression(),
      {}
    ),
    ( 'ARDRegression',
      ARDRegression(),
      {}  
    ),
    ( 'svm',
      svm.SVR(kernel='linear'),
      {}
    ),
    ( 'KNeighborsRegressor',
      KNeighborsRegressor(),
      {}
    )
  ]

  return models
