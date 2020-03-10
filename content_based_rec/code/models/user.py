import tensorflow as tf
import pandas as pd

users = [
         {"id": 1, "name": "Assad"  , "location":"limerick"},
         {"id": 2, "name": "Reezvee", "location": "Dublin"},
         {"id": 3, "name": "Shravan", "location": "Cork"},
         {"id": 4, "name": "Collins", "location": "404"}
        ]

def get_users():
  return list(pd.DataFrame(users).name)
