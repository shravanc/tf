import tensorflow as tf

print(tf.__version__)

a = tf.Variable([3, 3])
t2 = a
print(a)

b = tf.Variable([ [ [0., 1., 2.], [3., 4., 5.] ], [ [6., 7., 8.], [9., 10., 11.]]  ])
print(b)
print(b.numpy())

c = tf.Variable(42)
print(c)

# Reshape
#================================================================
r1 = tf.reshape(b, [2, 6])
print(r1.numpy())

r2 = tf.reshape(b, [1, 12])
print(r2.numpy())


# Ranking
#===============================================================
print(tf.rank(b))


t2 = tf.Variable([ [ [0., 1., 2.], [3., 4., 5.] ], [ [6., 7., 8.], [9., 10., 11.] ] ])

t3 = t2[1, 0, 2]
print(t3)



