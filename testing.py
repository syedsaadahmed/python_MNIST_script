import tensorflow as tf
hello = tf.constant('Hello, TensorFlow!')
sess = tf.Session()
sess.run(hello)
'Hello, TensorFlow!'
a = tf.constant(10)
b = tf.constant(32)
sess.run(a + b)
42
sess.close()