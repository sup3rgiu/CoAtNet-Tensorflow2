import tensorflow as tf

tfk = tf.keras
tfkl = tf.keras.layers


class PaddedConv2D(tfkl.Conv2D):
    def __init__(self, filters, kernel_size, padding=0, strides=1, name=None, **kwargs):
        self._padding = padding
        super().__init__(filters=filters, kernel_size=kernel_size, strides=strides, padding='valid', name=name, **kwargs)
        self.padding2d = tfkl.ZeroPadding2D(self._padding)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return super().call(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "padding": self._padding,
        })
        return config
    

class PaddedDepthwiseConv2D(tfkl.DepthwiseConv2D):
    def __init__(self, kernel_size, padding=0, strides=1, name=None, **kwargs):
        self._padding = padding
        super().__init__(kernel_size=kernel_size, strides=strides, padding='valid', name=name, **kwargs)
        self.padding2d = tfkl.ZeroPadding2D(self._padding)

    def call(self, inputs):
        x = self.padding2d(inputs)
        return super().call(x)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "padding": self._padding,
        })
        return config


def conv_3x3_bn_act(oup, downsample=False, bn_act=True, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name=None):
    stride = 2 if downsample else 1
    layers = []
    conv = PaddedConv2D(filters=oup, kernel_size=3, strides=stride, padding=1, use_bias=True, kernel_initializer=kernel_initializer, name=f'{name}.0')
    layers.append(conv)
    if bn_act:
        layers.append(tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{name}.1'))
        layers.append(tfkl.Activation('gelu', name=f'{name}.2'))
    return tfk.Sequential(layers, name=name)


class SE(tfkl.Layer):
    def __init__(self, filters, se_ratio=0.25, rd_channels=None,
                 kernel_initializer=tf.random_normal_initializer(stddev=0.02), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.filters = filters
        self.se_ratio = se_ratio
        self.rd_channels = rd_channels
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        axis = list(range(1, len(input_shape) - 1))
        
        rd_channels = self.rd_channels or self.filters
        rd_channels = int(rd_channels * self.se_ratio)
            
        #self.gap = tfkl.Lambda(lambda x: tf.reduce_mean(x, axis=axis, keepdims=True), name=f'{self.name}.avg_pool')
        self.gap = tfkl.GlobalAveragePooling2D(keepdims=True, name=f'{self.name}.avg_pool')
        self.fc = tfk.Sequential([
            #tfkl.Dense(units=rd_channels, use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.fc.0'),
            tfkl.Conv2D(filters=rd_channels, kernel_size=1, use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.fc.0'),
            tfkl.Activation('gelu', name=f'{self.name}.fc.1'),
            #tfkl.Dense(units=self.filters, use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.fc.2'),
            tfkl.Conv2D(filters=self.filters, kernel_size=1, use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.fc.2'),
            tfkl.Activation('sigmoid', name=f'{self.name}.fc.3')
        ], name=f'{self.name}.fc')

    def call(self, inputs):
        x = inputs
        x_shape = tf.shape(x)
        b, h, w, c = x_shape[0], x_shape[1], x_shape[2], x_shape[3]

        y = self.gap(x)
        y = self.fc(y)
        
        return x * y
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "se_ratio": self.se_ratio,
            "rd_channels": self.rd_channels
        })
        return config

 
class MBConv(tfkl.Layer):
    def __init__(self, oup, image_size, downsample=False, expansion=4, kernel_initializer=tf.random_normal_initializer(stddev=0.02), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.oup = oup
        self.image_size = image_size
        self.downsample = downsample
        self.expansion = expansion
        self.kernel_initializer = kernel_initializer
        
        self.expand_output = False # calculate expansion channels from output (vs input chs)
        
        # whether to apply downsampling in the Depthwise conv (vs in the first Conv2D)
        # see https://arxiv.org/pdf/2106.04803v2.pdf Appendix A.1, "Down-sampling" section, last paragraph
        # Authors suggests that "using stride-2 depthwise convolution is helpful but slower when model is small but not so much when model scales"
        self.downsample_dw = False

    def build(self, input_shape):
        stride = 1 if self.downsample == False else 2
        inp = input_shape[-1]
        hidden_dim = int((self.oup if self.expand_output else inp) * self.expansion)

        if self.downsample:
            #self.pool = tfkl.Lambda(lambda x: tf.nn.max_pool(x, ksize=3, strides=2, padding=[[0, 0], [1, 1], [1,  1], [0, 0]]), name=f'{self.name}.pool')
            self.pool = tfkl.MaxPool2D(pool_size=2, strides=2, padding='same', name=f'{self.name}.pool')
            self.proj = tfkl.Conv2D(filters=self.oup, kernel_size=1, strides=1, padding='valid', use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.proj')

        if self.expansion == 1:
            self.conv = tfk.Sequential([
                # depthwise
                PaddedConv2D(filters=hidden_dim, kernel_size=3, strides=stride, padding=1, groups=hidden_dim, use_bias=False, kernel_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.0'),
                tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}.conv.fn.1'),
                tfkl.Activation('gelu', name=f'{self.name}.conv.fn.2'),
                # expand
                tfkl.Conv2D(filters=self.oup, kernel_size=1, strides=1, padding='valid', use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.3'),
                #tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}.conv.fn.4'),
            ], name=f'{self.name}.conv.fn')
        else:
            self.conv = tfk.Sequential([
                # expand
                # downsample here if self.downsample_dw = False 
                tfkl.Conv2D(filters=hidden_dim, kernel_size=1, strides=1 if self.downsample_dw else stride, padding='valid', use_bias=False, kernel_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.0'),
                tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}.conv.fn.1'),
                tfkl.Activation('gelu', name=f'{self.name}.conv.fn.2'),
                # depthwise
                # downsample here if self.downsample_dw = True 
                PaddedDepthwiseConv2D(kernel_size=3, strides=stride if self.downsample_dw else 1, padding=1, use_bias=False, depthwise_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.3'), # PaddedConv2D(filters=hidden_dim, kernel_size=3, strides=stride if self.downsample_dw else 1, padding=1, groups=hidden_dim, use_bias=False, kernel_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.3'),
                tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}.conv.fn.4'),
                tfkl.Activation('gelu', name=f'{self.name}.conv.fn.5'),
                SE(filters=hidden_dim, rd_channels=self.oup if self.expand_output else None, se_ratio=0.25, kernel_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.6'),
                # shrink
                tfkl.Conv2D(filters=self.oup, kernel_size=1, strides=1, padding='valid', use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.conv.fn.7'),
                #tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}.conv.fn.8'),
            ], name=f'{self.name}.conv.fn')

        self.pre_norm = tfkl.BatchNormalization(momentum=0.9, epsilon=1e-5, name=f'{self.name}.conv.norm')
        
    def call(self, inputs, training=None):
        x = inputs
        if self.downsample:
            return self.proj(self.pool(x)) + self.conv(self.pre_norm(x, training=training), training=training)
        else:
            return x + self.conv(self.pre_norm(x, training=training), training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "oup": self.oup,
            "image_size": self.image_size,
            "downsample": self.downsample,
            "expansion": self.expansion
        })
        return config
    

class FeedForward(tfkl.Layer):
    def __init__(self, dim, hidden_dim, dropout=0., kernel_initializer=tf.random_normal_initializer(stddev=0.02), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        self.net = tfk.Sequential([
            #tfkl.Conv2D(filters=self.hidden_dim, kernel_size=1, kernel_initializer=self.kernel_initializer, name=f'{self.name}.net.0'),
            tfkl.Dense(units=self.hidden_dim, kernel_initializer=self.kernel_initializer, name=f'{self.name}.net.0'),
            tfkl.Activation('gelu', name=f'{self.name}.net.1'),
            tfkl.Dropout(self.dropout, name=f'{self.name}.net.2'),
            #tfkl.Conv2D(filters=self.dim, kernel_size=1, kernel_initializer=self.kernel_initializer, name=f'{self.name}.net.3'),
            tfkl.Dense(units=self.dim, kernel_initializer=self.kernel_initializer, name=f'{self.name}.net.3'),
            tfkl.Dropout(self.dropout, name=f'{self.name}.net.4'),
        ], name=f'{self.name}.net')

    def call(self, inputs, training=None):       
        return self.net(inputs, training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout
        })
        return config
    

class RelativeAttention(tfkl.Layer):
    def __init__(self,
                 oup,
                 image_size,
                 heads=None,
                 dim_head=32,
                 dropout=0.0,
                 kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                 name='attention',
                 **kwargs):
        super().__init__(name=name, **kwargs)

        self.oup = oup
        self.image_size = image_size
        self.heads = heads or oup // dim_head
        self.dim_head = dim_head
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer

        self.inner_dim = dim_head * self.heads
        self.ih, self.iw = image_size
        self.scale = dim_head ** -0.5       

    def build(self, input_shape):

        inp = input_shape[-1]
        self.project_out = not (self.heads == 1 and self.dim_head == inp)
     
        self.relative_bias_table = self.add_weight(
            name="relative_bias_table",
            shape=((2 * self.ih - 1) * (2 * self.iw - 1), self.heads),
            initializer=tf.zeros_initializer(),
            trainable=True
        )

        coords = tf.meshgrid(tf.range(self.ih), tf.range(self.iw))
        coords = tf.stack(coords) # 2, ih, iw
        coords = tf.reshape(coords, [2, -1]) # 2, ih*iw
        relative_coords = coords[:, :, None] - coords[:, None, :] # 2, ih*iw, ih*iw
        relative_coords = tf.transpose(relative_coords, [1,2,0])  # ih*iw, ih*iw, 2

        # i.e.:
        # relative_coords[:, :, 0] += self.ih - 1
        # relative_coords[:, :, 1] += self.iw - 1
        additive_values = tf.stack([
            tf.ones((relative_coords.shape[0], relative_coords.shape[1]), dtype=relative_coords.dtype) * (self.ih - 1),
            tf.ones((relative_coords.shape[0], relative_coords.shape[1]), dtype=relative_coords.dtype) * (self.iw - 1),
          ], axis=-1)
        relative_coords = relative_coords + additive_values # shift to start from 0

        # i.e.:
        # relative_coords[:, :, 0] *= 2 * self.iw - 1
        multiplicate_values = tf.stack([
            tf.ones((relative_coords.shape[0], relative_coords.shape[1]), dtype=relative_coords.dtype) * (2 * self.iw - 1),
            tf.ones((relative_coords.shape[0], relative_coords.shape[1]), dtype=relative_coords.dtype),
          ], axis=-1)
        relative_coords = relative_coords * multiplicate_values

        relative_index = tf.reduce_sum(relative_coords, -1) # ih*iw, ih*iw
        relative_index = tf.reshape(relative_index, -1) # ih*iw*ih*iw

        self.relative_index = tf.Variable(relative_index, trainable=False)

        #self.to_q = tfkl.Conv2D(self.inner_dim, kernel_size=1, use_bias=True, name=f'{self.name}.to_q', kernel_initializer=self.kernel_initializer)
        #self.to_k = tfkl.Conv2D(self.inner_dim, kernel_size=1, use_bias=True, name=f'{self.name}.to_k', kernel_initializer=self.kernel_initializer)
        #self.to_v = tfkl.Conv2D(self.inner_dim, kernel_size=1, use_bias=True, name=f'{self.name}.to_v', kernel_initializer=self.kernel_initializer)

        self.to_q = tfkl.Dense(self.inner_dim, use_bias=True, name=f'{self.name}.to_q', kernel_initializer=self.kernel_initializer)
        self.to_k = tfkl.Dense(self.inner_dim, use_bias=True, name=f'{self.name}.to_k', kernel_initializer=self.kernel_initializer)
        self.to_v = tfkl.Dense(self.inner_dim, use_bias=True, name=f'{self.name}.to_v', kernel_initializer=self.kernel_initializer)

        if self.project_out:
            self.to_out = tfkl.Dense(self.oup, name=f'{self.name}.to_out', kernel_initializer=self.kernel_initializer)

        self.dropout_layer = tfkl.Dropout(self.dropout)

    def call(self, x, training=None):
        x_shape = tf.shape(x)
        batch_size = x_shape[0]
        size_orig = x_shape[1:-1] # ih, iw   OR   ih*iw
        size = tf.reduce_prod(size_orig)  # ih*iw
        channels = x_shape[-1] # inp

        q = self.to_q(x) # bs, size, inner_dim
        k = self.to_k(x) # bs, size, inner_dim
        v = self.to_v(x) # bs, size, inner_dim

        q = tf.transpose(tf.reshape(q, shape=(batch_size, size, self.heads, self.dim_head)), perm=(0,2,1,3)) # bs, heads, size, dim_head
        k = tf.transpose(tf.reshape(k, shape=(batch_size, size, self.heads, self.dim_head)), perm=(0,2,1,3)) # bs, heads, size, dim_head
        v = tf.transpose(tf.reshape(v, shape=(batch_size, size, self.heads, self.dim_head)), perm=(0,2,1,3)) # bs, heads, size, dim_head

        q = q * self.scale
        k = tf.transpose(k, perm=(0, 1, 3, 2)) # bs, heads, dim_head, size
        attn = q @ k # bs, heads, size, size

        relative_bias = tf.gather(self.relative_bias_table, self.relative_index) # ih*iw*ih*iw, heads

        #i.e.: (h w) c -> 1 c h w
        relative_bias = tf.reshape(relative_bias, shape=(self.ih*self.iw, self.ih*self.iw, -1)) # ih*iw, ih*iw, heads
        relative_bias = tf.transpose(relative_bias, perm=(2, 0, 1)) # heads, ih*iw, ih*iw
        relative_bias = tf.expand_dims(relative_bias, axis=0) # 1, heads, ih*iw, ih*iw
        
        attn = attn + relative_bias # bs, heads, size, size

        attn = tf.nn.softmax(attn, axis=-1)
        attn = self.dropout_layer(attn, training=training)

        out = attn @ v # bs, heads, size, dim_head
        out = tf.transpose(out, perm=(0, 2, 1, 3)) # bs, size, heads, dim_head
        out = tf.reshape(out, shape=tf.concat([[batch_size], size_orig, [self.inner_dim]], axis=0)) # bs, *size_orig, inner_dim   (i.e. bs, *size_orig, heads*dim_head)

        if self.project_out:
            out = self.to_out(out) # bs, *size_orig, oup
            out = self.dropout_layer(out, training=training)
        return out

    def get_config(self):
        config = super().get_config()
        config.update({
            "oup": self.oup,
            "image_size": self.image_size,
            "heads": self.heads,
            "dim_head": self.dim_head,
            "dropout": self.dropout,
        })
        return config
    

class Transformer(tfkl.Layer):
    def __init__(self, oup, image_size, heads=None, dim_head=32, downsample=False, dropout=0., kernel_initializer=tf.random_normal_initializer(stddev=0.02), name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.oup = oup
        self.image_size = image_size
        self.heads = heads
        self.dim_head = dim_head
        self.downsample = downsample
        self.dropout = dropout
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):
        ih, iw = self.image_size
        hidden_dim = int(self.oup * 4)

        if self.downsample:
            #self.pool1 = tfkl.Lambda(lambda x: tf.nn.max_pool(x, ksize=3, strides=2, padding=[[0, 0], [1, 1], [1,  1], [0, 0]]), name=f'{self.name}.pool1')
            #self.pool2 = tfkl.Lambda(lambda x: tf.nn.max_pool(x, ksize=3, strides=2, padding=[[0, 0], [1, 1], [1,  1], [0, 0]]), name=f'{self.name}.pool2')
            self.pool1 = tfkl.MaxPool2D(pool_size=2, strides=2, padding='same', name=f'{self.name}.pool1')
            self.pool2 = tfkl.MaxPool2D(pool_size=2, strides=2, padding='same', name=f'{self.name}.pool2')
            #self.proj = tfkl.Conv2D(filters=self.oup, kernel_size=1, strides=1, padding='valid', use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.proj')
            self.proj = tfkl.Dense(units=self.oup, use_bias=True, kernel_initializer=self.kernel_initializer, name=f'{self.name}.proj')

        self.attn = RelativeAttention(self.oup, self.image_size, self.heads, self.dim_head, self.dropout, kernel_initializer=self.kernel_initializer, name=f'{self.name}.attn')
        self.ff = FeedForward(self.oup, hidden_dim, self.dropout, kernel_initializer=self.kernel_initializer, name=f'{self.name}.ff')

        self.pre_attn_layer_norm = tfkl.LayerNormalization(epsilon=1e-5, name=f'{self.name}.attn.1.norm')
        self.pre_ff_layer_norm = tfkl.LayerNormalization(epsilon=1e-5, name=f'{self.name}.ff.1.norm')

    def call(self, inputs, training=None):
        x = inputs
        if self.downsample:
            x = self.proj(self.pool1(x)) + self.attn(self.pool2(self.pre_attn_layer_norm(x, training=training)), training=training)
        else:
            x = x + self.attn(self.pre_attn_layer_norm(x, training=training), training=training)
        x = x + self.ff(self.pre_ff_layer_norm(x, training=training), training=training)
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "oup": self.oup,
            "image_size": self.image_size,
            "heads": self.heads,
            "dim_head": self.dim_head,
            "downsample": self.downsample,
            "dropout": self.dropout,
        })
        return config
    

def CoAtNet(image_size, in_channels, num_blocks, channels, num_classes=1000, block_types=['C', 'C', 'T', 'T'], activation='softmax', name="CoAtNet", seed=None):
    ih, iw = image_size
    block = {'C': MBConv, 'T': Transformer}

    if seed:
        tf.random.set_seed(seed)
        
    kernel_initializer = tf.random_normal_initializer(stddev=0.02)

    def _make_layer(block, oup, depth, image_size, kernel_initializer, name, **kwargs):
        layers = []
        for i in range(depth):
            if i == 0:
                layers.append(block(oup, image_size, downsample=True, kernel_initializer=kernel_initializer, name=f'{name}.{i}', **kwargs))
            else:
                layers.append(block(oup, image_size, downsample=False, kernel_initializer=kernel_initializer, name=f'{name}.{i}', **kwargs))
        return tfk.Sequential(layers, name=name)

    def _make_stem(block, oup, depth, kernel_initializer, name, **kwargs):
        layers = []
        for i in range(depth):
            bn_act = i < depth - 1 # do not add BatchNorm + Act in the last block
            if i == 0:
                layers.append(block(oup, downsample=True, bn_act=bn_act, kernel_initializer=kernel_initializer, name=f'{name}.{i}', **kwargs))
            else:
                layers.append(block(oup, downsample=False, bn_act=bn_act, kernel_initializer=kernel_initializer, name=f'{name}.{i}', **kwargs))
        return tfk.Sequential(layers, name=name)
        
    s0 = _make_stem(
        conv_3x3_bn_act, channels[0], num_blocks[0], kernel_initializer, name='s0') # Stem stage
    s1 = _make_layer(
        block[block_types[0]], channels[1], num_blocks[1], (ih // 4, iw // 4), kernel_initializer, name='s1')
    s2 = _make_layer(
        block[block_types[1]], channels[2], num_blocks[2], (ih // 8, iw // 8), kernel_initializer, name='s2')
    s3 = _make_layer(
        block[block_types[2]], channels[3], num_blocks[3], (ih // 16, iw // 16), kernel_initializer, name='s3')
    s4 = _make_layer(
        block[block_types[3]], channels[4], num_blocks[4], (ih // 32, iw // 32), kernel_initializer, name='s4')
    
    last_norm = tfkl.LayerNormalization(epsilon=1e-5, name='norm')
    gap = tfkl.GlobalAveragePooling2D(name='pool')
    fc = tfkl.Dense(units=num_classes, use_bias=True, kernel_initializer=kernel_initializer, name='logits')

    input_shape = (ih, iw, in_channels)
    x = tfkl.Input(input_shape, name='input')

    h = x
    h = s0(h)
    h = s1(h)
    h = s2(h)
    h = s3(h)
    h = s4(h)

    h = last_norm(h)
    h = gap(h)
    h = fc(h)
    
    if activation != None:
        h = tfkl.Activation(activation, dtype='float32', name='last_activation')(h)

    return tfk.Model(x, h, name=name)


def coatnet_0(image_size=(224, 224), in_channels=3, num_classes=1000, activation='softmax', seed=None, **kwargs):
    num_blocks = [2, 2, 3, 5, 2]            # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet(image_size, in_channels, num_blocks, channels, num_classes=num_classes, activation=activation, seed=seed, **kwargs)


def coatnet_1(image_size=(224, 224), in_channels=3, num_classes=1000, activation='softmax', seed=None, **kwargs):
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [64, 96, 192, 384, 768]      # D
    return CoAtNet(image_size, in_channels, num_blocks, channels, num_classes=num_classes, activation=activation, seed=seed, **kwargs)


def coatnet_2(image_size=(224, 224), in_channels=3, num_classes=1000, activation='softmax', seed=None, **kwargs):
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [128, 128, 256, 512, 1026]   # D
    return CoAtNet(image_size, in_channels, num_blocks, channels, num_classes=num_classes, activation=activation, seed=seed, **kwargs)


def coatnet_3(image_size=(224, 224), in_channels=3, num_classes=1000, activation='softmax', seed=None, **kwargs):
    num_blocks = [2, 2, 6, 14, 2]           # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet(image_size, in_channels, num_blocks, channels, num_classes=num_classes, activation=activation, seed=seed, **kwargs)


def coatnet_4(image_size=(224, 224), in_channels=3, num_classes=1000, activation='softmax', seed=None, **kwargs):
    num_blocks = [2, 2, 12, 28, 2]          # L
    channels = [192, 192, 384, 768, 1536]   # D
    return CoAtNet(image_size, in_channels, num_blocks, channels, num_classes=num_classes, activation=activation, seed=seed, **kwargs)
