import numpy as np
import tensorflow as tf

from models.user    import users, get_users
from models.movies  import movies, get_movies_fetures, get_movies
from models.ratings  import ratings, get_user_movies #, test
from models.genres  import genres, get_features

num_users   = len(users)
num_movies  = len(movies)
num_feats   = len(genres)
num_recommendations = 2

users     = get_users()
movies    = get_movies()
features  = get_features()

print("----")
print("----", users)
print("----", movies)
print("----", features)
print(num_users)
print(num_movies)
print(num_feats)
print("----")


users_movies   = tf.constant(get_user_movies(), dtype=tf.float32)
movies_feats  = tf.constant(get_movies_fetures(), dtype=tf.float32)
print("user_items----->",users_movies)
print("item_feats----->",movies_feats)

users_feats = tf.matmul(users_movies, movies_feats)
print("itermediate_state----->", users_feats)


users_feats = users_feats/tf.reduce_sum(users_feats,axis=1,keepdims=True)
print("users_feats---->", users_feats)



top_users_features = tf.nn.top_k(users_feats, num_feats)[1]
print("top_users_features---->", top_users_features)



for i in range(num_users):
    print("--------------inside lop-------------", i)
    print("--------------inside lop-------------", features)
    print("--------------inside lop-------------", top_users_features[i])
    feature_names = [features[int(index)] for index in top_users_features[i]]
    print('{}: {}'.format(users[i],feature_names))


users_ratings = tf.matmul(users_feats,tf.transpose(movies_feats))
print(users_ratings)



users_ratings_new = tf.where(tf.equal(users_movies, tf.zeros_like(users_movies)),
                                  users_ratings,
                                  tf.zeros_like(tf.cast(users_movies, tf.float32)))
print(users_ratings_new)



top_movies = tf.nn.top_k(users_ratings_new, num_recommendations)[1]
print(top_movies)


for i in range(num_users):
    movie_names = [movies[index] for index in top_movies[i]]
    print('{}: {}'.format(users[i],movie_names))


