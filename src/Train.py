#keep compatability among different python version
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import os

import GAN

# import VAE

#Define our input and output data
#load all augmented depth images and labels
train_images = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/data/NYU_Image.npy')
train_images = train_images.reshape([-1,128,128])
print('train_images.shape: ', train_images.shape)

train_labels = np.load('/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/data/NYU_Label.npy')
train_labels = train_labels.reshape([-1,42])
print('train_labels.shape: ', train_labels.shape)


#calculate training time
def elapsed(sec):
    if sec < 60:
        return str(sec) + " sec"
    elif sec < (60 * 60):
        return str(sec / 60) + " min"
    else:
        return str(sec / (60 * 60)) + " hr"


tf.reset_default_graph()

epoches = 50
batch_size = 100

#Inject depth images and joints to GAN
Noise = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='Noise')

X_in_image = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128], name='X_in_image')
X_in_label = tf.placeholder(dtype=tf.float32, shape=[None, 42], name='X_in_label')



'''
GAN: we need to generate the images like real image
'''
DepthGAN = GAN.DepthGAN()

#use joints to build the discriminator and generator
g = DepthGAN.generator(X_in_label, reuse=False)
d_real, _, hidden_layer, x_Norm = DepthGAN.discriminator(X_in_image)
d_fake, _, g_hidden_layer, _ = DepthGAN.discriminator(g, reuse=True)

dis_joints = DepthGAN.posterior_recognition(hidden_layer)

vars_g = [var for var in tf.trainable_variables() if var.name.startswith("generator")]
vars_d = [var for var in tf.trainable_variables() if var.name.startswith("discriminator")]
vars_d_joints = [var for var in tf.trainable_variables() if var.name.startswith("posterior")]

d_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d)
g_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_g)
d_joints_reg = tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1e-6), vars_d_joints)


'''
# loss
# real image set to 1
# fake image set to 0
'''

loss_dis = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_real, labels=tf.ones_like(d_real))) + \
         tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.zeros_like(d_fake)))
loss_joints_pos =  tf.reduce_mean(tf.reduce_sum(tf.squared_difference(dis_joints, X_in_label), 1))
loss_d_sum = loss_dis + loss_joints_pos + d_reg + d_joints_reg   # + loss_smooth

# loss_gen = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=d_fake, labels=tf.ones_like(d_fake)))
loss_gen = tf.reduce_mean(tf.abs(d_fake - tf.ones_like(d_fake)))
loss_rescon =  tf.reduce_mean(tf.reduce_sum(tf.clip_by_value(tf.squared_difference(g, x_Norm), 0, 1), 1))
loss_g_sum = loss_gen + loss_rescon + g_reg  # + loss_smooth


# set AdamOptimizer to decrease the generator and discriminator losses
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.control_dependencies(update_ops):  #smooth_loss +

    optimizer_d = tf.train.RMSPropOptimizer(learning_rate=0.001).\
        minimize(loss_d_sum, var_list=vars_d+vars_d_joints)  ##+ loss_dis_joints + d_joints_reg

    optimizer_g = tf.train.RMSPropOptimizer(learning_rate=0.001).\
        minimize(loss_g_sum, var_list=vars_g)#

'''
Training process
'''
steps = []
gen_loss_list = []
dis_loss_list = []

path = '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/results/'

'''
Start session and initialize all the variables
'''

init = tf.global_variables_initializer()

saver = tf.train.Saver()

start_time_sum = time.time()

with tf.Session() as sess:

    sess.run(init)

    for epoch in range(epoches):

        idx = np.random.randint(0, train_labels.shape[0], train_labels.shape[0])
        images = train_images[idx]
        labels = train_labels[idx]

        for step in range(0, train_labels.shape[0] // batch_size):

            # input depth image to GAN
            batch_image = images[step * batch_size : (step + 1) * batch_size, ]
            batch_joint = labels[step * batch_size : (step + 1) * batch_size, ]


            g_ls, d_ls = sess.run([loss_g_sum, loss_d_sum], feed_dict={X_in_image: batch_image, X_in_label: batch_joint})  #, Noise: batch_noise

            start_time = time.time()

            sess.run(optimizer_d, feed_dict={X_in_image: batch_image, X_in_label: batch_joint})

            sess.run(optimizer_g, feed_dict={X_in_image: batch_image, X_in_label: batch_joint})

            duration = time.time() - start_time

            print("Epoch: %d, Step: %d,  Dis_loss: %f, Gen_loss: %f, Duration:%f sec"
                  % (epoch, step, d_ls, g_ls, duration))

            #save the loss of generator and discriminator
            steps.append(epoch * train_labels.shape[0] // batch_size + step)
            gen_loss_list.append(g_ls)
            dis_loss_list.append(d_ls)

            #show the last batch of images
            if not step % 200:

                gen_img = sess.run(g, feed_dict={X_in_label: batch_joint})

                gen_img = gen_img.reshape([-1,128,128])

                r, c = 2, 2
                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(gen_img[cnt, :], cmap='gray')
                        axs[i, j].axis('off')
                        cnt += 1
                fig.savefig(os.path.join(path + 'gen_images/', '%d_%d.png' % (epoch, step)))
                plt.close()

                fig, axs = plt.subplots(r, c)
                cnt = 0
                for i in range(r):
                    for j in range(c):
                        axs[i, j].imshow(batch_image[cnt, :], cmap='gray')
                        axs[i, j].axis('off')
                        cnt += 1
                fig.savefig(os.path.join(path + 'raw_images/', '%d_%d.png' % (epoch, step)))
                plt.close()

    saver.save(sess, '/media/xc/c728432c-8ae3-4aeb-b43d-9ef02faac4f8/VAE_GAN/tmp/model.ckpt')

duration_time_sum = time.time() - start_time_sum

print("The total training time: ",elapsed(duration_time_sum))


'''
#show the loss of generaor and discriminator at every batch
'''
fig = plt.figure(figsize=(8,6))
plt.plot(steps, gen_loss_list, label='gen_loss')
plt.plot(steps, dis_loss_list, label='dis_loss')

plt.xlabel('Step')
plt.ylabel('Loss')
plt.title('The loss of train')
plt.legend()
plt.legend(loc = 'upper right')
plt.savefig(os.path.join(path, 'loss_curve.png'))
plt.show()




















