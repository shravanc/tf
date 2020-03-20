import pandas as pd
from models.genres import genres


movies = [
          {"id": 1, "title": "Inception"          , "genres": [1, 2, 5] },
          {"id": 2, "title": "Blood Diamond"      , "genres": [1, 2]    },
          {"id": 3, "title": "Catch me If you can", "genres": [3, 4]    },
          {"id": 4, "title": "Departed"           , "genres": [1, 3, 4] },
          {"id": 5, "title": "The Revenent"       , "genres": [5]       },
          {"id": 6, "title": "Shutter Island"     , "genres": [1, 5]    },
         ]


def get_movies():
  return list(pd.DataFrame(movies).title)

def get_movies_fetures():
  df = pd.DataFrame(movies)
  
  features = []
  for index, item in df.iterrows():
    data = [0] * len(genres)
    for i in item.genres:
      data[i-1] = 1
    features.append(data)
  
  print("FEATURES************", features)
   
  return features 
