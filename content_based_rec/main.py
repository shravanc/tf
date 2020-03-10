import numpy as np
import tensorflow as tf

#tf.enable_eager_execution()
print(tf.__version__)


users = ['Ryan', 'Danielle',  'Vijay', 'Chris']
movies = ['Star Wars', 'The Dark Knight', 'Shrek', 'The Incredibles', 'Bleu', 'Memento']
features = ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']

num_users = len(users)
num_movies = len(movies)
num_feats = len(features)
num_recommendations = 2

print("----", users)
print("----", movies)
print("----", features)
print(num_users)
print(num_movies)
print(num_feats)
print("----")

# each row represents a user's rating for the different movies
users_movies = tf.constant([
                [4,  6,  8,  0, 0, 0],
                [0,  0, 10,  0, 8, 3],
                [0,  6,  0,  0, 3, 7],
                [10, 9,  0,  5, 0, 2]],dtype=tf.float32)

# features of the movies one-hot encoded
# e.g. columns could represent ['Action', 'Sci-Fi', 'Comedy', 'Cartoon', 'Drama']
movies_feats = tf.constant([
                [1, 1, 0, 0, 1],
                [1, 1, 0, 0, 0],
                [0, 0, 1, 1, 0],
                [1, 0, 1, 1, 0],
                [0, 0, 0, 0, 1],
                [1, 0, 0, 0, 1]],dtype=tf.float32)



users_feats = tf.matmul(users_movies,movies_feats)
print(users_feats)


users_feats = users_feats/tf.reduce_sum(users_feats,axis=1,keepdims=True)
print(users_feats)



top_users_features = tf.nn.top_k(users_feats, num_feats)[1]
print(top_users_features)


for i in range(num_users):
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



