from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import minmax_scale
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import PolynomialFeatures

def get_distributions(X):

  distributions = [
    ('nscaled_data', X),
    ('standard_scaling',
        StandardScaler().fit_transform(X)),
    ('minmax_scaling',
        MinMaxScaler().fit_transform(X)),
    ('maxabs_scaling',
        MaxAbsScaler().fit_transform(X)),
    ('robust_scaling',
        RobustScaler(quantile_range=(25, 75)).fit_transform(X)),
    ('transformation_Yeo_Johnson',
     PowerTransformer(method='yeo-johnson').fit_transform(X)),
    ('transformation_gaussian_pdf',
        QuantileTransformer(output_distribution='normal')
        .fit_transform(X)),
    ('transformation_uniform_pdf',
        QuantileTransformer(output_distribution='uniform')
        .fit_transform(X)),
    ('samplewise_L2_normalizing',
        Normalizer().fit_transform(X)),
  ]

  return distributions


