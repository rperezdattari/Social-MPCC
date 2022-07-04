import tensorflow as tf
import tensorflow_addons as tfa
from datetime import datetime
import os


class NeuralNetwork:
    def __init__(self, policy_learning_rate, load_policy, dim_a, network_loc,network_load_path, image_width):
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_width  # we assume that the image is a square
        self.dim_a = dim_a
        self.today = datetime.now()
        self.network_loc = network_loc
        self.network_loc += '_{}-{}-{}-{}-{}/'.format(self.today.year, self.today.month, self.today.day, self.today.hour, self.today.minute)
        self.network_load_path = network_load_path
        self.lstm_hidden_state_size = 150
        self.feature_extractor = 'vgg'  # Options: vgg, resnet

        if self.feature_extractor == 'vgg':
            self.features_dim = 512
        elif self.feature_extractor == 'resnet':
            self.features_dim = 2048

        # Build Neural Network
        self._build_feature_extraction()
        self._build_policy()

        # Save initialized models to overwrite previous
        self.save_policy()

        # Load models
        if load_policy:
            self._load_policy()

        # Initialize optimizers
        self.policy_optimizer = tfa.optimizers.AdamW(learning_rate=policy_learning_rate, weight_decay=0.0001)
        #self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)

    def _build_feature_extraction(self):
        transition_model_input = tf.keras.layers.Input(shape=(self.image_width, self.image_width, 3),
                                                       name='transition_model_input')

        if self.feature_extractor == 'vgg':
            self.NN_feature_extraction = tf.keras.applications.VGG19(include_top=False,
                                                                     weights='imagenet',
                                                                     input_tensor=transition_model_input,
                                                                     input_shape=(self.image_width, self.image_width, 3),
                                                                     pooling='avg')
        elif self.feature_extractor == 'resnet':
            self.NN_feature_extraction = tf.keras.applications.ResNet50V2(include_top=False,
                                                                          weights='imagenet',
                                                                          input_tensor=transition_model_input,
                                                                          input_shape=(self.image_width, self.image_width, 3),
                                                                          pooling='avg')

    def _build_policy(self):
        # Inputs
        state_representation_input = tf.keras.layers.Input(shape=(None, self.features_dim), batch_size=None, name='state_representation_input')
        lstm_hidden_state_in = [tf.keras.layers.Input(shape=self.lstm_hidden_state_size, name='lstm_hidden_state_h_in'),
                                tf.keras.layers.Input(shape=self.lstm_hidden_state_size, name='lstm_hidden_state_c_in')]
        velocity_input = tf.keras.layers.Input(shape=(None, 1), batch_size=None, name='velocity_input')
        drive_input = tf.keras.layers.Input(shape=(None, 1), batch_size=None, name='drive_input')

        # Fully connected layers
        state_representation_input_norm = tf.keras.layers.LayerNormalization(name='norm_state_representation_input')(state_representation_input)
        fc_1 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_1'))(state_representation_input_norm)
        fc_1_norm = tf.keras.layers.LayerNormalization(name='norm_fc_1')(fc_1)

        fc_2 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_2'))(velocity_input)
        fc_2_norm = tf.keras.layers.LayerNormalization(name='norm_fc_2')(fc_2)

        fc_3 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(100, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_3'))(drive_input)
        fc_3_norm = tf.keras.layers.LayerNormalization(name='norm_fc_3')(fc_3)

        # Concat 1
        concat_1 = tf.concat([fc_1_norm, fc_2_norm, fc_3_norm], name='concat_1', axis=1)
        #concat_1 = tf.concat([fc_1, fc_2, fc_3], name='concat_1', axis=1)
        fc_concat_1 = tf.keras.layers.Dense(self.lstm_hidden_state_size, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_concat_1')(concat_1)
        fc_concat_1_norm = tf.keras.layers.LayerNormalization(name='fc_concat_1_norm')(fc_concat_1)

        # Init LSTM
        cell = tf.keras.layers.LSTMCell(self.lstm_hidden_state_size)
        lstm = tf.keras.layers.RNN(cell, return_sequences=False, return_state=True, name='rnn_layer')

        # RNN layer
        _, lstm_hidden_state_h_out, lstm_hidden_state_c_out = lstm(inputs=fc_concat_1_norm, initial_state=lstm_hidden_state_in)
        #_, lstm_hidden_state_h_out, lstm_hidden_state_c_out = lstm(inputs=fc_concat_1, initial_state=lstm_hidden_state_in)

        # Concat 2
        #concat_2 = tf.concat([fc_concat_1_norm[:, -1, :], lstm_hidden_state_h_out], name='concat_2', axis=1)
        #concat_2 = tf.concat([fc_concat_1[:, -1, :], lstm_hidden_state_h_out], name='concat_2', axis=1)
        concat_2 = lstm_hidden_state_h_out

        # More fully connected layers
        fc_4 = tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_4')(concat_2)
        fc_4_norm = tf.keras.layers.LayerNormalization(name='norm_fc_4')(fc_4)

        #fc_5 = tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_5')(fc_4)
        #fc_5_norm = tf.keras.layers.LayerNormalization(name='norm_fc_5')(fc_5)

        fc_6 = tf.keras.layers.Dense(self.dim_a, activation='linear', name='fc_6')(fc_4_norm)
        policy_output = fc_6

        # Create model
        self.NN_policy = tf.keras.Model(inputs=[state_representation_input, lstm_hidden_state_in, velocity_input, drive_input],
                                        outputs=[policy_output, [lstm_hidden_state_h_out, lstm_hidden_state_c_out]],
                                        name='policy')
        self.NN_policy.summary()

    def save_policy(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)
        print("Saving model in: " + self.network_loc + 'policy')
        self.NN_policy.save_weights(self.network_loc + 'policy')

    def _load_policy(self):
        print("Loading model from: " + self.network_load_path + '/policy')
        self.NN_policy.load_weights(self.network_load_path + '/policy')
