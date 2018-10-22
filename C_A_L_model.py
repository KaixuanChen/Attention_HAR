import tensorflow as tf
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell
from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import rnn_decoder
from tensorflow.python.ops.distributions.normal import Normal


def _weight_variable(shape):
    initial = tf.truncated_normal(shape=shape, stddev=0.01)
    return tf.Variable(initial)


def _bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv2d(x, w):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_3x3(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='SAME')


def max_pool_2x3(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 2, 3, 1], strides=[1, 2, 3, 1], padding='SAME')


def max_pool_1x1(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

def _log_likelihood(loc_means, locs, variance):
    loc_means = tf.stack(loc_means)  # [timesteps, batch_sz, loc_dim]
    locs = tf.stack(locs)
    gaussian = Normal(loc_means, variance)
    logll = gaussian._log_prob(x=locs)  # [timesteps, batch_sz, loc_dim]
    logll = tf.reduce_sum(logll, 2)
    return tf.transpose(logll)  # [batch_sz, timesteps]


class RetinaSensor(object):
    # one scale
    def __init__(self, img_size_width, img_size_height, patch_window_width, patch_window_height):
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height
        self.patch_window_width = patch_window_width
        self.patch_window_height = patch_window_height

    def __call__(self, img_ph, loc):
        img = tf.reshape(img_ph, [
            tf.shape(img_ph)[0],
            self.img_size_width,
            self.img_size_height,
            1
        ])
        '''
        tf.image.extract_glimpse:
        If the windows only partially
        overlaps the inputs, the non overlapping areas will be filled with
        random noise.
        '''
        pth = tf.image.extract_glimpse(

            img, # input
            [self.patch_window_width, self.patch_window_height], # size
            loc) # offset
        # pth: [tf.shape(img_ph)[0], patch_window_width, patch_window_height, 1]

        return tf.reshape(pth, [tf.shape(loc)[0],
                                self.patch_window_width * self.patch_window_height])


class GlimpseNetwork(object):
    def __init__(self, img_size_width, img_size_height,
                 patch_window_width, patch_window_height,
                 loc_dim, g_size, l_size, output_size):
        self.retina_sensor = RetinaSensor(img_size_width, img_size_height,
                                          patch_window_width, patch_window_height)

        # layer 1
        self.g1_w = _weight_variable((patch_window_width * patch_window_height, g_size))
        self.g1_b = _bias_variable((g_size,))
        self.l1_w = _weight_variable((loc_dim, l_size))
        self.l1_b = _bias_variable((l_size,))
        # layer 2
        self.g2_w = _weight_variable((g_size, output_size))
        self.g2_b = _bias_variable((output_size,))
        self.l2_w = _weight_variable((l_size, output_size))
        self.l2_b = _bias_variable((output_size,))

    def __call__(self, imgs_ph, locs):
        pths = self.retina_sensor(imgs_ph, locs)

        g = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(pths, self.g1_w, self.g1_b)),
                            self.g2_w, self.g2_b)
        l = tf.nn.xw_plus_b(tf.nn.relu(tf.nn.xw_plus_b(locs, self.l1_w, self.l1_b)),
                            self.l2_w, self.l2_b)

        return tf.nn.relu(g + l)


class LocationNetwork(object):
    def __init__(self, loc_dim, rnn_output_size, variance=0.22, is_sampling=False):
        self.loc_dim = loc_dim  # 2, (x,y)
        self.variance = variance
        self.w = _weight_variable((rnn_output_size, loc_dim))
        self.b = _bias_variable((loc_dim,))

        self.is_sampling = is_sampling

    def __call__(self, cell_output):
        mean = tf.nn.xw_plus_b(cell_output, self.w, self.b)
        mean = tf.clip_by_value(mean, -1., 1.)
        mean = tf.stop_gradient(mean)

        if self.is_sampling:
            loc = mean + tf.random_normal(
                (tf.shape(cell_output)[0], self.loc_dim),
                stddev=self.variance)
            loc = tf.clip_by_value(loc, -1., 1.)
        else:
            loc = mean
        loc = tf.stop_gradient(loc)
        return loc, mean

class CNN(object):
    def __init__(self, img_size_width, img_size_height,
                 CNN_patch_width, CNN_patch_height, CNN_patch_number):
        self.img_size_width = img_size_width
        self.img_size_height = img_size_height
        self.CNN_patch_width = CNN_patch_width
        self.CNN_patch_height = CNN_patch_height
        self.CNN_patch_number = CNN_patch_number

    def __call__(self, imgs_ph):
        imgs_ph = tf.reshape(imgs_ph, [-1, self.img_size_height, self.img_size_width, 1])
        W_conv1 = _weight_variable(
            [self.CNN_patch_width, self.CNN_patch_height, 1, self.CNN_patch_number]
        )
        # patch width x height, in size 1, out size patch_nb
        b_conv1 = _bias_variable([self.CNN_patch_number])
        h_conv1 = tf.nn.relu(conv2d(imgs_ph, W_conv1) + b_conv1)

        W_fc1 = _weight_variable([self.img_size_height * self.img_size_width
                                  * self.CNN_patch_number,
                                  self.img_size_height * self.img_size_width])
        b_fc1 = _bias_variable([self.img_size_height * self.img_size_width])
        h_pool2_flat = tf.reshape(h_conv1,
                                  [-1,
                                   self.img_size_height * self.img_size_width
                                   * self.CNN_patch_number]
                                  )
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
        # return tf.reshape(h_fc1, [-1, self.img_size_height, self.img_size_width, 1])
        return h_fc1

class RecurrentAttentionModel(object):
    def __init__(self, img_size_width, img_size_height,
                 CNN_patch_width, CNN_patch_height, CNN_patch_number,
                 patch_window_width, patch_window_height, g_size, l_size, glimpse_output_size,
                 loc_dim, variance,
                 cell_size, num_glimpses, num_classes,
                 learning_rate, learning_rate_decay_factor, min_learning_rate, training_batch_num,
                 max_gradient_norm, last_lstm_size, n_time_window,
                 is_training=False):

        self.img_ph = tf.placeholder(tf.float32, [None, img_size_width * img_size_height])
        self.lbl_ph = tf.placeholder(tf.int64, [None])

        self.global_step = tf.Variable(0, trainable=False)
        # decayed_learning_rate = learning_rate * decay_rate ^ (global_step / training_batch_num)
        self.learning_rate = tf.maximum(tf.train.exponential_decay(
            learning_rate, self.global_step,
            training_batch_num, # batch number
            learning_rate_decay_factor,
            # If the argument staircase is True,
            # then global_step / decay_steps is an integer division
            # and the decayed learning rate follows a staircase function.
            staircase=True),
            min_learning_rate)

        cell = BasicLSTMCell(cell_size)

        with tf.variable_scope('CNN'):
            cnn_network = CNN(img_size_width, img_size_height,
                              CNN_patch_width, CNN_patch_height, CNN_patch_number)

        with tf.variable_scope('GlimpseNetwork'):
            glimpse_network = GlimpseNetwork(img_size_width,
                                             img_size_height,
                                             patch_window_width,
                                             patch_window_height,
                                             loc_dim,
                                             g_size,
                                             l_size,
                                             glimpse_output_size)
        with tf.variable_scope('LocationNetwork'):
            location_network = LocationNetwork(loc_dim=loc_dim,
                                               rnn_output_size=cell.output_size, # cell_size
                                               variance=variance,
                                               is_sampling=is_training)

        # Core Network
        self.img_ph = cnn_network(self.img_ph)
        batch_size = tf.shape(self.img_ph)[0]  # training_batch_size * M
        init_loc = tf.random_uniform((batch_size, loc_dim), minval=-1, maxval=1)
        # shape: (batch_size, loc_dim), range: [-1,1)
        init_state = cell.zero_state(batch_size, tf.float32)

        init_glimpse = glimpse_network(self.img_ph, init_loc)
        rnn_inputs = [init_glimpse]
        rnn_inputs.extend([0] * num_glimpses)

        self.locs, loc_means = [], []

        def loop_function(prev, _):
            loc, loc_mean = location_network(prev)
            self.locs.append(loc)
            loc_means.append(loc_mean)
            glimpse = glimpse_network(self.img_ph, loc)
            return glimpse

        rnn_outputs, _ = rnn_decoder(rnn_inputs, init_state, cell, loop_function=loop_function)

        # Time independent baselines
        with tf.variable_scope('Baseline'):
            baseline_w = _weight_variable((cell.output_size, 1))
            baseline_b = _bias_variable((1,))
        baselines = []
        for output in rnn_outputs[1:]:
            baseline = tf.nn.xw_plus_b(output, baseline_w, baseline_b)
            baseline = tf.squeeze(baseline)
            baselines.append(baseline)
        baselines = tf.stack(baselines)  # [timesteps, batch_sz]
        baselines = tf.transpose(baselines)  # [batch_sz, timesteps]

        # Classification. Take the last step only.
        rnn_last_output = rnn_outputs[-1]
        with tf.variable_scope('Classification'):
            logit_w = _weight_variable((cell.output_size, num_classes))
            logit_b = _bias_variable((num_classes,))
        logits = tf.nn.xw_plus_b(rnn_last_output, logit_w, logit_b)
        self.prediction = tf.argmax(logits, 1)
        self.softmax = tf.nn.softmax(logits)

        with tf.variable_scope('LSTM_Classification'):
            last_lstm_w_in = _weight_variable((cell.output_size, last_lstm_size))
            last_lstm_b_in = _bias_variable((last_lstm_size,))
            last_lstm_in = tf.matmul(rnn_last_output, last_lstm_w_in) + last_lstm_b_in
            last_lstm_in = tf.reshape(last_lstm_in, [-1, n_time_window, last_lstm_size])

            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                cell = tf.nn.rnn_cell.BasicLSTMCell(last_lstm_size, forget_bias=1.0, state_is_tuple=True)
            else:
                cell = tf.contrib.rnn.BasicLSTMCell(last_lstm_size)
            # lstm cell is divided into two parts (c_state, h_state)
            init_state_last_lstm = cell.zero_state(batch_size // n_time_window, dtype=tf.float32)
            lstm_outputs, final_state = tf.nn.dynamic_rnn(cell, last_lstm_in,
                                                     initial_state=init_state_last_lstm, time_major=False)
            last_lstm_w_out = _weight_variable((cell.output_size, num_classes))
            last_lstm_b_out = _bias_variable((num_classes,))

            if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
                lstm_outputs = tf.unpack(tf.transpose(lstm_outputs, [1, 0, 2]))  # states is the last outputs
            else:
                lstm_outputs = tf.unstack(tf.transpose(lstm_outputs, [1, 0, 2]))
            lstm_logits = tf.matmul(lstm_outputs[-1], last_lstm_w_out) + last_lstm_b_out
            lstm_logits = tf.reshape(tf.tile(lstm_logits, (1, n_time_window)), [-1, num_classes])
            self.lstm_prediction = tf.argmax(lstm_logits, 1)
            self.lstm_softmax = tf.nn.softmax(lstm_logits)



        if is_training:
            # classification loss
            self.cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=logits))
            self.lstm_cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.lbl_ph, logits=lstm_logits))
            # RL reward
            reward = tf.cast(tf.equal(self.prediction, self.lbl_ph), tf.float32)
            rewards = tf.expand_dims(reward, 1)  # [batch_sz, 1]
            rewards = tf.tile(rewards, (1, num_glimpses))  # [batch_sz, timesteps]
            advantages = rewards - tf.stop_gradient(baselines)
            self.advantage = tf.reduce_mean(advantages)
            logll = _log_likelihood(loc_means, self.locs, variance)
            logllratio = tf.reduce_mean(logll * advantages)
            self.reward = tf.reduce_mean(reward)
            # baseline loss
            self.baselines_mse = tf.reduce_mean(tf.square((rewards - baselines)))
            # hybrid loss
            self.loss = -logllratio + self.cross_entropy + self.baselines_mse + self.lstm_cross_entropy
            params = tf.trainable_variables()
            gradients = tf.gradients(self.loss, params)
            clipped_gradients, norm = tf.clip_by_global_norm(gradients, max_gradient_norm)
            self.train_op = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(
                zip(clipped_gradients, params), global_step=self.global_step)

        self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=99999999)