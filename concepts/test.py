#test.py
import tensorflow as tf

#allow growth to take up minimal resources
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

sess = tf.compat.v1.Session(config=config)

