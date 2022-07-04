import tensorflow as tf
import os


class NeuralNetwork:
    def __init__(self, policy_learning_rate, transition_model_learning_rate, lstm_hidden_state_size,
                 load_policy, dim_a, network_loc, image_size):
        self.lstm_hidden_state_size = lstm_hidden_state_size
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_size  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc
        self.transition_model_learning_rate = transition_model_learning_rate
        self.feature_extractor = 'vgg'  # Options: vgg, resnet

        if self.feature_extractor == 'vgg':
            self.features_dim = 512
        elif self.feature_extractor == 'resnet':
            self.features_dim = 2048

        # Build Neural Network
        self._build_feature_extraction()
        self._build_policy()

        # Save initialized models to overwrite previous
        self.save_policy(id='0')

        # Load models
        if load_policy:
            self._load_policy()

        # Initialize optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)

    def _build_feature_extraction(self):
        transition_model_input_raw = tf.keras.layers.Input(shape=(self.image_width, self.image_width, 3),
                                                           name='transition_model_input')

        if self.feature_extractor == 'vgg':
            #transition_model_input = tf.keras.applications.vgg19.preprocess_input(transition_model_input_raw)
            self.NN_feature_extraction = tf.keras.applications.VGG19(include_top=False,
                                                                     weights='imagenet',
                                                                     input_tensor=transition_model_input_raw,
                                                                     input_shape=(self.image_width, self.image_width, 3),
                                                                     pooling='avg')
        elif self.feature_extractor == 'resnet':
            transition_model_input = tf.keras.applications.resnet.preprocess_input(transition_model_input_raw)
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

        policy_input = tf.keras.layers.LayerNormalization(name='norm_policy')(state_representation_input)

        # Fully connected layer
        fc_5 = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(self.lstm_hidden_state_size, activation=tf.keras.layers.LeakyReLU(alpha=0.01)), name='fc_5')(policy_input)
        fc_5_norm = tf.keras.layers.LayerNormalization(name='norm_fc_5')(fc_5)

        # LSTM
        cell = tf.keras.layers.LSTMCell(self.lstm_hidden_state_size)
        lstm = tf.keras.layers.RNN(cell, return_sequences=False, return_state=True, name='rnn_layer')

        # Init RNN layer
        _, lstm_hidden_state_h_out, lstm_hidden_state_c_out = lstm(inputs=fc_5_norm, initial_state=lstm_hidden_state_in)

        # Fully connected layers
        lstm_hidden_state_h_out_norm = tf.keras.layers.LayerNormalization(name='lstm_hidden_state_h_out_norm')(lstm_hidden_state_h_out)
        fc_6 = tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_6')(lstm_hidden_state_h_out_norm)
        fc_6_norm = tf.keras.layers.LayerNormalization(name='norm_fc_6')(fc_6)
        fc_7 = tf.keras.layers.Dense(self.dim_a, activation='linear', name='fc_7')(fc_6_norm)
        policy_output = fc_7

        # Create model
        self.NN_policy = tf.keras.Model(inputs=[state_representation_input, lstm_hidden_state_in],
                                        outputs=[policy_output, [lstm_hidden_state_h_out, lstm_hidden_state_c_out]],
                                        name='policy')
        self.NN_policy.summary()

    def save_policy(self, id):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)
        print("Saving model in: " + self.network_loc + '_policy_' + id)
        self.NN_policy.save_weights(self.network_loc + '_policy_' + id)

    def _load_policy(self, id):
        print("Loading model from: " + self.network_loc + '_policy_' + id)
        self.NN_policy.load_weights(self.network_loc + '_policy_' + id)
