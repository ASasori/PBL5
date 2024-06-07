# --------- Define auxiliary classes ---------

import os
import keras
import tensorflow as tf

# --------- Define auxiliary classes ---------
import keras
import tensorflow as tf
from keras.layers import GaussianNoise
from tensorflow.keras import backend as K


@keras.saving.register_keras_serializable(package="1DCNN_Transformer")
class ECA(tf.keras.layers.Layer):
    def __init__(self, kernel_size=5, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.kernel_size = kernel_size
        self.conv = tf.keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="same", use_bias=False)

    def call(self, inputs, mask=None):
        nn = tf.keras.layers.GlobalAveragePooling1D()(inputs, mask=mask)
        nn = tf.expand_dims(nn, -1)
        nn = self.conv(nn)
        nn = tf.squeeze(nn, -1)
        nn = tf.nn.sigmoid(nn)
        nn = nn[:,None,:]
        return inputs * nn

    def get_config(self):
        base_config = super().get_config()
        config = {
            # "supports_masking" : keras.saving.serialize_keras_object(self.supports_masking),
            "kernel_size" : keras.saving.serialize_keras_object(self.kernel_size)
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls,config):
        kernel_size_config = config.pop("kernel_size")
        kernel_size = keras.saving.deserialize_keras_object(kernel_size_config)
        return cls(kernel_size, **config)

@keras.saving.register_keras_serializable(package="1DCNN_Transformer")
class LateDropout(tf.keras.layers.Layer):
    def __init__(self, rate, noise_shape=None, start_step=0, **kwargs):
        super().__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.noise_shape = noise_shape
        self.start_step = start_step
        self.dropout = tf.keras.layers.Dropout(rate, noise_shape=noise_shape)

    def build(self, input_shape):
        super().build(input_shape)
        agg = tf.VariableAggregation.ONLY_FIRST_REPLICA
        self._train_counter = tf.Variable(0, dtype="int64", aggregation=agg, trainable=False)

    def call(self, inputs, training=False):
        x = tf.cond(self._train_counter < self.start_step, lambda:inputs, lambda:self.dropout(inputs, training=training))
        if training:
            self._train_counter.assign_add(1)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            # "supports_masking" : keras.saving.serialize_keras_object(self.supports_masking),
            "rate" : keras.saving.serialize_keras_object(self.rate),
            "start_step" : keras.saving.serialize_keras_object(self.start_step),
            "noise_shape" : keras.saving.serialize_keras_object(self.noise_shape),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls,config):
        rate_config = config.pop("rate")
        rate = keras.saving.deserialize_keras_object(rate_config)
        start_step_config = config.pop("start_step")
        start_step = keras.saving.deserialize_keras_object(start_step_config)
        noise_shape_config = config.pop("noise_shape")
        noise_shape = keras.saving.deserialize_keras_object(noise_shape_config)
        return cls(rate, noise_shape, start_step,  **config)

@keras.saving.register_keras_serializable(package="1DCNN_Transformer")
class CausalDWConv1D(tf.keras.layers.Layer):
    def __init__(self,
        kernel_size=17,
        dilation_rate=1,
        use_bias=False,
        depthwise_initializer='glorot_uniform',
        name='', **kwargs):
        super().__init__(name=name,**kwargs)
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.use_bias = use_bias
        self.depthwise_initializer=depthwise_initializer
        self.lname=name

        self.causal_pad = tf.keras.layers.ZeroPadding1D((dilation_rate*(kernel_size-1),0),name=name + '_pad')
        self.dw_conv = tf.keras.layers.DepthwiseConv1D(
                            kernel_size,
                            strides=1,
                            dilation_rate=dilation_rate,
                            padding='valid',
                            use_bias=use_bias,
                            depthwise_initializer=depthwise_initializer,
                            name=name + '_dwconv')
        self.supports_masking = True

    def call(self, inputs):
        x = self.causal_pad(inputs)
        x = self.dw_conv(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "kernel_size" : keras.saving.serialize_keras_object(self.kernel_size),
            "dilation_rate" : keras.saving.serialize_keras_object(self.dilation_rate),
            "use_bias" : keras.saving.serialize_keras_object(self.use_bias),
            "depthwise_initializer" : keras.saving.serialize_keras_object(self.depthwise_initializer),
            "name" : keras.saving.serialize_keras_object(self.lname),
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls,config):
        kernel_size_config = config.pop("kernel_size")
        kernel_size = keras.saving.deserialize_keras_object(kernel_size_config)
        dilation_rate_config = config.pop("dilation_rate")
        dilation_rate = keras.saving.deserialize_keras_object(dilation_rate_config)
        bias_config = config.pop("use_bias")
        bias = keras.saving.deserialize_keras_object(bias_config)
        depthwise_config = config.pop("depthwise_initializer")
        depthwise = keras.saving.deserialize_keras_object(depthwise_config)
        name_config = config.pop("name")
        name = keras.saving.deserialize_keras_object(name_config)

        return cls(kernel_size,dilation_rate,bias,depthwise,name, **config)

def Conv1DBlock(channel_size,
          kernel_size,
          dilation_rate=1,
          drop_rate=0.0,
          expand_ratio=2,
          se_ratio=0.25,
          activation='swish',
          name=None):
    '''
    efficient conv1d block, @hoyso48
    '''
    if name is None:
        name = str(tf.keras.backend.get_uid("mbblock"))
    # Expansion phase
    def apply(inputs):
        channels_in = tf.keras.backend.int_shape(inputs)[-1]
        channels_expand = channels_in * expand_ratio

        skip = inputs

        x = tf.keras.layers.Dense(
            channels_expand,
            use_bias=True,
            activation=activation,
            name=name + '_expand_conv')(inputs)

        # Depthwise Convolution
        x = CausalDWConv1D(kernel_size,
            dilation_rate=dilation_rate,
            use_bias=False,
            name=name + '_dwconv')(x)

        x = tf.keras.layers.BatchNormalization(momentum=0.95, name=name + '_bn')(x)

        x  = ECA()(x)

        x = tf.keras.layers.Dense(
            channel_size,
            use_bias=True,
            name=name + '_project_conv')(x)

        if drop_rate > 0:
            x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1), name=name + '_drop')(x)

        if (channels_in == channel_size):
            x = tf.keras.layers.add([x, skip], name=name + '_add')
        return x

    return apply


@keras.saving.register_keras_serializable(package="1DCNN_Transformer")
class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, dim=256, num_heads=4, dropout=0, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.scale = self.dim ** -0.5
        self.num_heads = num_heads
        self.dropout = dropout
        self.qkv = tf.keras.layers.Dense(3 * dim, use_bias=False)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.proj = tf.keras.layers.Dense(dim, use_bias=False)
        self.supports_masking = True

    def call(self, inputs, mask=None):
        qkv = self.qkv(inputs)
        qkv = tf.keras.layers.Permute((2, 1, 3))(tf.keras.layers.Reshape((-1, self.num_heads, self.dim * 3 // self.num_heads))(qkv))
        q, k, v = tf.split(qkv, [self.dim // self.num_heads] * 3, axis=-1)

        attn = tf.matmul(q, k, transpose_b=True) * self.scale

        if mask is not None:
            mask = mask[:, None, None, :]

        attn = tf.keras.layers.Softmax(axis=-1)(attn, mask=mask)
        attn = self.drop1(attn)

        x = attn @ v
        x = tf.keras.layers.Reshape((-1, self.dim))(tf.keras.layers.Permute((2, 1, 3))(x))
        x = self.proj(x)
        return x

    def get_config(self):
        base_config = super().get_config()
        config = {
            "dim" : self.dim,
            "num_heads" : self.num_heads,
            "dropout" : self.dropout,
        }
        return {**base_config, **config}

    @classmethod
    def from_config(cls,config):
        dim_config = config.pop("dim")
        dim = keras.saving.deserialize_keras_object(dim_config)
        num_heads_config = config.pop("num_heads")
        num_heads = keras.saving.deserialize_keras_object(num_heads_config)
        dropout_config = config.pop("dropout")
        dropout = keras.saving.deserialize_keras_object(dropout_config)
        return cls(dim,num_heads,dropout)

def TransformerBlock(dim=256, num_heads=4, expand=4, attn_dropout=0.2, drop_rate=0.2, activation='swish'):
    def apply(inputs):
        x = inputs
        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = MultiHeadSelfAttention(dim=dim,num_heads=num_heads,dropout=attn_dropout)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([inputs, x])
        attn_out = x

        x = tf.keras.layers.BatchNormalization(momentum=0.95)(x)
        x = tf.keras.layers.Dense(dim*expand, use_bias=False, activation=activation)(x)
        x = tf.keras.layers.Dense(dim, use_bias=False)(x)
        x = tf.keras.layers.Dropout(drop_rate, noise_shape=(None,1,1))(x)
        x = tf.keras.layers.Add()([attn_out, x])
        return x
    return apply

class GaussianNoiseLayer(tf.keras.layers.Layer):
    def __init__(self, noise_intensity, **kwargs):
        super(GaussianNoiseLayer, self).__init__(**kwargs)
        self.noise_intensity = noise_intensity

    def call(self, inputs, training=None):
        if training:
            noise = K.random_normal(shape=tf.shape(inputs), mean=0., stddev=self.noise_intensity)
            return inputs + noise
        return inputs

    def get_config(self):
        config = super(GaussianNoiseLayer, self).get_config()
        config.update({"noise_intensity": self.noise_intensity})
        return config
n_frames = 30
MAX_LEN = n_frames # number of frame
CHANNELS = 258 # number of keypoint value
NUM_CLASSES = 20
PAD = -100

# ----------------------------------------- DEFINE MODEL -----------------------------
def get_model(max_len=MAX_LEN, dropout_step=0, dim=256):
    inp = tf.keras.Input((max_len,CHANNELS))
    x = GaussianNoiseLayer(0.05)(inp)
    x = tf.keras.layers.Masking(mask_value=PAD,input_shape=(max_len,CHANNELS))(x) #we don't need masking layer with inference
    # x = inp
    ksize = 7
    # x = tf.keras.layers.Permute((2,1))(x)
    x = tf.keras.layers.Dense(dim, use_bias=False,name='stem_conv')(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.95,name='stem_bn')(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
    x = TransformerBlock(dim,expand=2)(x)

    if dim == 384: #for the 4x sized model
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = Conv1DBlock(dim,ksize,drop_rate=0.2)(x)
        x = TransformerBlock(dim,expand=2)(x)

    x = tf.keras.layers.Dense(dim*2,activation=None,name='top_conv')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    # x = LateDropout(0.5, start_step=dropout_step)(x)
    x = tf.keras.layers.Dense(NUM_CLASSES,name='classifier',activation="softmax")(x)
    return tf.keras.Model(inp, x)

def load_model(path='1DCNN_Transformer_F30_K7_0406_nonpermute.weights.h5'):
    model = get_model()
    module_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(module_dir,path)
    model.load_weights(model_path)
    return model
