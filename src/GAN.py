import tensorflow as tf


class DepthGAN(object):

    # def __init__(self):
    #     return

    # Input(-1, 42)->Output(-1, 128, 128, 1)
    def generator(self, joints_in, reuse=None):  #, noise_in

        with tf.variable_scope('generator', reuse=reuse):

            n = 32

            joints = tf.reshape(joints_in, [-1, 42])

            input_map = tf.layers.batch_normalization(tf.layers.dense(joints, units=4 * 4 * 32, activation=tf.nn.leaky_relu))

            x_in = tf.reshape(input_map, shape=[-1, 4, 4, 32])

            # noise(-1, 42)->(-1,4 * 4 * 32)
            x = tf.layers.batch_normalization(tf.layers.dense(x_in, units=4 * 4 * 32, activation=tf.nn.leaky_relu))

            # image(-1,4,4,32)->(-1,8,8,32)
            gen1 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(x,
                                                                            filters=n, kernel_size=6, strides=2,
                                                                            use_bias=True,
                                                                            kernel_initializer=tf.truncated_normal_initializer(
                                                                                stddev=0.01),
                                                                            padding='same', activation=tf.nn.leaky_relu))

            # image(-1,8,8,32)->(-1,16,16,32)
            gen2 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen1,
                                                                            filters=n, kernel_size=6, strides=2,
                                                                            use_bias=True,
                                                                            kernel_initializer=tf.truncated_normal_initializer(
                                                                                stddev=0.01),
                                                                            padding='same', activation=tf.nn.leaky_relu))


            # image(-1,16,16,32)->(-1,32,32,32)
            gen3 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen2,
                                                                            filters=n, kernel_size=6, strides=2,
                                                                            use_bias=True,
                                                                            kernel_initializer=tf.truncated_normal_initializer(
                                                                                stddev=0.01),
                                                                            padding='same', activation=tf.nn.leaky_relu))

            # image(-1,32,32,32)->(1, 64, 64, 32)
            gen4 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen3,
                                                                            filters=n, kernel_size=6, strides=2,
                                                                            use_bias=True,
                                                                            kernel_initializer=tf.truncated_normal_initializer(
                                                                                stddev=0.01),
                                                                            padding='same', activation=tf.nn.leaky_relu))

            # image(1, 64, 64, 32)->(1, 128, 128, 1)
            gen5 = tf.layers.batch_normalization(tf.layers.conv2d_transpose(gen4,
                                                                            filters=1, kernel_size=6, strides=2,
                                                                            use_bias=True,
                                                                            kernel_initializer=tf.truncated_normal_initializer(
                                                                                stddev=0.01),
                                                                            padding='same', activation=tf.nn.tanh))

            return gen5


    # Input(-1, 128, 128)->Output(-1, 0/1)
    def discriminator(self, image, reuse=None):

        with tf.variable_scope('discriminator', reuse=reuse):

            n = 32

            x_Norm = tf.layers.batch_normalization(tf.reshape(image, shape=[-1, 128, 128, 1]))

            # original image(128,128,1)->(64,64,32)
            dis_conv1 = tf.layers.batch_normalization(tf.layers.conv2d(x_Norm, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            # image size(64,64,32)->(32,32,32)
            dis_conv2 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv1, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            # # to be later used for recognition or other discriminative purpose
            dis_hidden_layer = dis_conv2
            #
            # image size(32,32,32)->(16,16,32)
            dis_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv2, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            # image size(16,16,32)->(8,8,32)
            dis_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv3, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            # image size(8,8,32)->(4,4,32)
            dis_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv4, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))
            flat = tf.layers.flatten(dis_conv5)

            x = tf.layers.dense(flat, 128)

            d_out_logits = tf.layers.dense(x, 1)

            d_out = tf.nn.sigmoid(d_out_logits)

            return d_out, d_out_logits, dis_hidden_layer, x_Norm


    #from the shape of hiddde layer is [-1,32,32,32] to the joints of size [-1, 42]
    def posterior_recognition(self, dis_hidden_layer=None, reuse=None):

        with tf.variable_scope('posterior', reuse=reuse):

            n =  32

            # image size(32,32,32)->(16,16,32)
            dis_conv3 = tf.layers.batch_normalization(tf.layers.conv2d(dis_hidden_layer, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            # image size(16,16,32)->(8,8,32)
            dis_conv4 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv3, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            # image size(8,8,32)->(4,4,32)
            dis_conv5 = tf.layers.batch_normalization(tf.layers.conv2d(dis_conv4, filters=n, kernel_size=6, strides=2, padding="same",
                                                      activation=tf.nn.leaky_relu))

            flat = tf.layers.flatten(dis_conv5)

            x = tf.layers.dense(flat, 32 * 4 * 4)

            d_out_joint = tf.layers.dense(x, 42)

            return d_out_joint








































