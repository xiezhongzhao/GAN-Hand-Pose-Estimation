import tensorflow as tf
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import GAN
import VAE

batch_size = 64


train_images = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/data/NYU_Image.npy')
train_images = train_images.reshape([-1,128,128])
print('train_images.shape: ', train_images.shape)

train_labels = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/data/NYU_Label.npy')
train_labels = train_labels.reshape([-1,42])
print('train_labels.shape: ', train_labels.shape)


X_in_image = tf.placeholder(dtype=tf.float32, shape=[None,128,128], name='X_in_image')
X_in_joint = tf.placeholder(dtype=tf.float32, shape=[None,train_labels.shape[1]], name='X_in_joint')


# input joints to encoder of VAE
batch_joint = train_labels[0 : batch_size, ]
# input depth image to GAN
batch_image = train_images[0 : batch_size, ]

DepthGAN = GAN.DepthGAN()

g = DepthGAN.generator(batch_joint)
g = tf.reshape(g, [-1, 128, 128])

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

    sess.run(init)

    saver.restore(sess, "/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/tmp/model.ckpt")

    for i in range(0, 1):

        # input joints to encoder of VAE
        batch_joint = train_labels[i * batch_size : (i + 1) * batch_size, ]
        # input depth image to GAN
        batch_image = train_images[i * batch_size : (i + 1) * batch_size, ]


        # _, loss = sess.run([optimizer_g, img_gan_loss], feed_dict={X_in_image: batch_image, X_in_noise: batch_joint})
        gen_img = sess.run(g, feed_dict={X_in_joint: batch_joint})

        # print("gen_img.shape: ", gen_img[0].shape)

        plt.imshow(gen_img[0], cmap='gray')
        plt.show()




















