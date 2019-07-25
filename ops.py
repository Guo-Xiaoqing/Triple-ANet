import tensorflow as tf
import tensorflow.contrib as tf_contrib
import numpy as np

# Xavier : tf_contrib.layers.xavier_initializer()
# He : tf_contrib.layers.variance_scaling_initializer()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf_contrib.layers.l2_regularizer(0.0001)

weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
weight_regularizer = None

##################################################################################
# Layer
##################################################################################

def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w),
                             strides=[1, stride, stride, 1], padding='VALID')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            #x = tf.layers.conv2d(inputs=x, filters=channels,
            #                     kernel_size=kernel, kernel_initializer=weight_init,
            #                     kernel_regularizer=weight_regularizer,
            #                     strides=stride, use_bias=use_bias)
            x = tf.contrib.layers.conv2d(inputs=x, num_outputs=channels, kernel_size=kernel, 
                                         stride=stride, padding='VALID',
                                         activation_fn=None,
                                         weights_initializer=tf.contrib.layers.xavier_initializer())

        return x


def atrous_conv2d(x, channels, kernel=3, rate=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.atrous_conv2d(value=x, filters=spectral_norm(w), rate=2, padding='SAME')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :             
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.atrous_conv2d(value=x, filters=w, rate=2, padding='SAME')

    return x
 

def atrous_pool2d(x, channels, kernel=3, rate=2, pad=0, pad_type='zero', use_bias=True, sn=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        if sn :
            w = tf.constant("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.atrous_conv2d(value=x, filters=spectral_norm(w), rate=2, padding='SAME')
            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :             
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape()[-1], channels], initializer=weight_init,
                                regularizer=weight_regularizer)
            x = tf.nn.atrous_conv2d(value=x, filters=w, rate=2, padding='SAME')

    return x


def deconv(x, channels, kernel=4, stride=2, padding='SAME', use_bias=True, sn=False, scope='deconv_0'):
    with tf.variable_scope(scope):
        x_shape = x.get_shape().as_list()

        if padding == 'SAME':
            output_shape = [x_shape[0], x_shape[1] * stride, x_shape[2] * stride, channels]

        else:
            output_shape =[x_shape[0], x_shape[1] * stride + max(kernel - stride, 0), x_shape[2] * stride + max(kernel - stride, 0), channels]

        if sn :
            w = tf.get_variable("kernel", shape=[kernel, kernel, channels, x.get_shape()[-1]], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d_transpose(x, filter=spectral_norm(w), output_shape=output_shape, strides=[1, stride, stride, 1], padding=padding)

            if use_bias :
                bias = tf.get_variable("bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else :
            x = tf.layers.conv2d_transpose(inputs=x, filters=channels,
                                           kernel_size=kernel, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer,
                                           strides=stride, padding=padding, use_bias=use_bias)

        return x

def fully_conneted(x, units, use_bias=True, sn=False, scope='fully_0'):
    with tf.variable_scope(scope):
        x = flatten(x)
        shape = x.get_shape().as_list()
        channels = shape[-1]

        if sn :
            w = tf.get_variable("kernel", [channels, units], tf.float32,
                                     initializer=weight_init, regularizer=weight_regularizer)
            if use_bias :
                bias = tf.get_variable("bias", [units],
                                       initializer=tf.constant_initializer(0.0))

                x = tf.matmul(x, spectral_norm(w)) + bias
            else :
                x = tf.matmul(x, spectral_norm(w))

        else :
            x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def flatten(x) :
    return tf.layers.flatten(x)

def hw_flatten(x) :
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])


#########################
#deformable conv
#########################

# Definition of the regular 2D convolutional
def deform_conv(x, kernel_size, stride, output_channals, mode):
    if mode == 'offset':
        layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=stride, padding='SAME', kernel_initializer = tf.zeros_initializer(), bias_initializer = tf.zeros_initializer())
        layer_output = tf.clip_by_value(layer_output, -0.25*int(x.shape[1]), 0.25*int(x.shape[1]))
    if mode == 'weight':
        layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=stride, padding='SAME', bias_initializer = tf.zeros_initializer())
    if mode == 'feature':
        #layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=kernel_size, padding='SAME', kernel_initializer = tf.constant_initializer(0.5), bias_initializer = tf.zeros_initializer())   
        #layer_output = tf.layers.conv2d(x, filters=output_channals, kernel_size=kernel_size, strides=kernel_size, padding='SAME', initializer=weight_init,regularizer=weight_regularizer)
        layer_output = conv(x, output_channals, kernel=kernel_size, stride=kernel_size, sn=True, scope='feature')
    return layer_output

# Create the pn [1, 1, 1, 2N]
def get_pn(kernel_size, dtype):
    pn_x, pn_y = np.meshgrid(range(-(kernel_size-1)//2, (kernel_size-1)//2+1), range(-(kernel_size-1)//2, (kernel_size-1)//2+1), indexing="ij")

    # The order is [x1, x2, ..., y1, y2, ...]
    pn = np.concatenate((pn_x.flatten(), pn_y.flatten()))

    pn = np.reshape(pn, [1, 1, 1, 2 * kernel_size ** 2])

    # Change the dtype of pn
    pn = tf.constant(pn, dtype)

    return pn

# Create the p0 [1, h, w, 2N]
def get_p0(kernel_size, x_size, dtype):

    bs, h, w, C = x_size

    p0_x, p0_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
    p0_x = p0_x.flatten().reshape(1, h, w, 1).repeat(kernel_size ** 2, axis=3)
    p0_y = p0_y.flatten().reshape(1, h, w, 1).repeat(kernel_size ** 2, axis=3)
    p0 = np.concatenate((p0_x, p0_y), axis=3)

    # Change the dtype of p0
    p0 = tf.constant(p0, dtype)

    return p0

def get_q(x_size, dtype):

    bs, h, w, c = x_size

    q_x, q_y = np.meshgrid(range(0, h), range(0, w), indexing="ij")
    q_x = q_x.flatten().reshape(h, w, 1)
    q_y = q_y.flatten().reshape(h, w, 1)
    q = np.concatenate((q_x, q_y), axis=2)

    # Change the dtype of q
    q = tf.constant(q, dtype)

    return q

def reshape_x_offset(x_offset, kernel_size):

    bs, h, w, N, C = x_offset.get_shape().as_list()

    # Get the new_shape
    new_shape = [bs, h, w * kernel_size, C]
    x_offset = [tf.reshape(x_offset[:, :, :, s:s+kernel_size, :], new_shape) for s in range(0, N, kernel_size)]
    x_offset = tf.concat(x_offset, axis=2)

    # Reshape to final shape [batch_size, h*kernel_size, w*kernel_size, C]
    x_offset = tf.reshape(x_offset, [bs, h * kernel_size, w * kernel_size, C])

    return x_offset

def deform_con2v(input, num_outputs, kernel_size, stride, trainable, name, reuse):
    N = kernel_size ** 2 
    with tf.variable_scope(name, reuse=reuse):
        bs, h, w, C = input.get_shape().as_list()
        
        # offset with shape [batch_size, h, w, 2N]
        offset = deform_conv(input, kernel_size, stride, 2 * N, "offset")
        #offset = tf.constant(0.0,shape=[bs, h, w, 2*N])
        # delte_weight with shape [batch_size, h, w, N * C]
        #delte_weight = deform_conv(input, kernel_size, stride, N * C, "weight")
        #delte_weight = tf.sigmoid(delte_weight)

        # pn with shape [1, 1, 1, 2N]
        pn = get_pn(kernel_size, offset.dtype)

        # p0 with shape [1, h, w, 2N]
        p0 = get_p0(kernel_size, [bs, h, w, C], offset.dtype)

        # p with shape [batch_size, h, w, 2N]
        p = pn + p0 + offset

        # Reshape p to [batch_size, h, w, 2N, 1, 1]
        p = tf.reshape(p, [bs, h, w, 2 * N, 1, 1])

        # q with shape [h, w, 2]
        q = get_q([bs, h, w, C], offset.dtype)

        # Bilinear interpolation kernel G ([batch_size, h, w, N, h, w])
        gx = tf.maximum(1 - tf.abs(p[:, :, :, :N, :, :] - q[:, :, 0]), 0)
        gy = tf.maximum(1 - tf.abs(p[:, :, :, N:, :, :] - q[:, :, 1]), 0)
        G = gx * gy

        # Reshape G to [batch_size, h*w*N, h*w]
        G = tf.reshape(G, [bs, h * w * N, h * w])

        # Reshape x to [batch_size, h*w, C]
        x = tf.reshape(input, [bs, h*w, C])

        # x_offset with shape [batch_size, h, w, N, C]
        x = tf.reshape(tf.matmul(G, x), [bs, h, w, N, C])

        # Reshape x_offset to [batch_size, h*kernel_size, w*kernel_size, C]
        x = reshape_x_offset(x, kernel_size)

        # Reshape delte_weight to [batch_size, h*kernel_size, w*kernel_size, C]
        #delte_weight = tf.reshape(delte_weight, [batch_size, h*kernel_size, w*kernel_size, C])

        #y = x_offset * delte_weight

        # Get the output of the deformable convolutional layer
        x = deform_conv(x, kernel_size, stride, num_outputs, "feature")

    return x, offset    

##################################################################################
# Sampling
##################################################################################
def make_png(att, scale):
    att_current = up_sample_bilinear(att, scale_factor=scale)
    att_current = tf.nn.relu(att_current)
    att_current = tf.reduce_mean(att_current,axis=-1)
    att_current = tf.stack([att_current, att_current, att_current])
    att_current = tf.transpose(att_current, perm=[1, 2, 3, 0])
    return att_current

def global_avg_pooling(x):
    gap = tf.reduce_mean(x, axis=[1, 2])

    return gap

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [np.int32(h * scale_factor), np.int32(w * scale_factor)]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def up_sample_bilinear(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [np.int32(h * scale_factor), np.int32(w * scale_factor)]
    return tf.image.resize_bilinear(x, size=new_size)

def up_sample_bicubic(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [np.int32(h * scale_factor), np.int32(w * scale_factor)]
    return tf.image.resize_bicubic(x, size=new_size)
##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

##################################################################################
# Normalization function
##################################################################################

def batch_norm(x, is_training=True, scope='batch_norm'):
    #return tf.layers.batch_normalization(x, training=is_training)
    return tf_contrib.layers.batch_norm(x,decay=0.9, epsilon=1e-05,
                                        center=True, scale=True, updates_collections=tf.GraphKeys.UPDATE_OPS,
                                        is_training=is_training, scope=scope)

def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# Loss function
##################################################################################

def class_loss(class_logits, label, num_class):
    loss = 0
    loss = tf.losses.softmax_cross_entropy(tf.one_hot(label, num_class), class_logits, weights=1.0)

    return loss