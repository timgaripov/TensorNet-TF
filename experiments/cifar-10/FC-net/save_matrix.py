import tensorflow as tf
import net
import numpy as np

net.build()
vars = tf.all_variables()

saver = tf.train.Saver()
sess = tf.Session()

print([var.name for var in vars])

mat = list(filter(lambda x: x.name == 'linear_1/weights:0', vars))[0]
saver.restore(sess, './log/checkpoint-40000')

W = sess.run(mat)
print(W.shape)
np.savez('mat.npz', mat=W)
