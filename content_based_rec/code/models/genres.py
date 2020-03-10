import pandas as pd

genres =  [
            {"id": 1, "title": "sci-fi"   , "movie_id": 1},
            {"id": 2, "title": "drama"    , "movie_id": 2},
            {"id": 3, "title": "thriller" , "movie_id": 3},
            {"id": 4, "title": "action"   , "movie_id": 4},
            {"id": 5, "title": "adventure", "movie_id": 5},
          ]

def get_features():
  return list(pd.DataFrame(genres).title)
