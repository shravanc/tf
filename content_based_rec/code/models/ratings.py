import tensorflow as tf
import pandas as pd
from models.movies import movies

ratings =  [
            {"user_id": 1, "item_id": 1, "rating": 4},
            {"user_id": 1, "item_id": 2, "rating": 6},
            {"user_id": 1, "item_id": 3, "rating": 8},
            
            {"user_id": 2, "item_id": 3, "rating": 10},
            {"user_id": 2, "item_id": 5, "rating": 8},
            {"user_id": 2, "item_id": 6, "rating": 3},
            
            {"user_id": 3, "item_id": 2, "rating": 6},
            {"user_id": 3, "item_id": 5, "rating": 3},
            {"user_id": 3, "item_id": 6, "rating": 7},
            
            {"user_id": 4, "item_id": 1, "rating": 10},
            {"user_id": 4, "item_id": 2, "rating": 9},
            {"user_id": 4, "item_id": 5, "rating": 5},
            {"user_id": 4, "item_id": 6, "rating": 2},
          ]

  

def get_user_movies():
  features = pd.DataFrame(ratings).groupby('user_id')
  totla_movies = len(movies)

  user_ratings = []
  for user_id, df in features:
    data = [0] * totla_movies
    for index, item in df.iterrows():
      data[item.item_id-1] =  item.rating
      
    user_ratings.append(data)
   
  return user_ratings 
