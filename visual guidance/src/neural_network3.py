import tensorflow as tf
import os


class NeuralNetwork:
    def __init__(self, policy_learning_rate, load_policy, dim_a, network_loc, image_width):
        self.policy_learning_rate = policy_learning_rate
        self.image_width = image_width  # we assume that the image is a square
        self.dim_a = dim_a
        self.network_loc = network_loc

        # Build Neural Network
        self._build_policy()

        # Save initialized models to overwrite previous
        self.save_policy()

        # Load models
        if load_policy:
            self._load_policy()

        # Initialize optimizers
        self.policy_optimizer = tf.keras.optimizers.Adam(learning_rate=policy_learning_rate)

    def _build_policy(self):
        # Inputs
        velocity_input = tf.keras.layers.Input(shape=1, batch_size=None, name='velocity_input')
        observation_input = tf.keras.layers.Input(shape=(self.image_width, self.image_width, 3),
                                                  batch_size=None,
                                                  name='observation_input')

        # Convolutional layers
        conv1 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', name='conv1')(observation_input)
        conv1_norm = tf.keras.layers.LayerNormalization(name='conv1_norm')(conv1)

        conv2 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', name='conv2')(conv1_norm)
        conv2_norm = tf.keras.layers.LayerNormalization(name='conv2_norm')(conv2)

        conv3 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=(2, 2), activation='relu', name='conv3')(conv2_norm)
        conv3_norm = tf.keras.layers.LayerNormalization(name='conv3_norm')(conv3)

        # Flatten tensor
        conv3_norm_flatten = tf.keras.layers.Flatten(name='conv3_norm_flatten')(conv3_norm)

        # Fully connected layers
        fc_1 = tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_1')(conv3_norm_flatten)
        fc_1_norm = tf.keras.layers.LayerNormalization(name='norm_fc_1')(fc_1)

        fc_2 = tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_2')(velocity_input)
        fc_2_norm = tf.keras.layers.LayerNormalization(name='norm_fc_2')(fc_2)

        concat = tf.concat([fc_1_norm, fc_2_norm], name='concat', axis=1)

        fc_3 = tf.keras.layers.Dense(1000, activation=tf.keras.layers.LeakyReLU(alpha=0.01), name='fc_3')(concat)
        fc_3_norm = tf.keras.layers.LayerNormalization(name='norm_fc_3')(fc_3)

        fc_4 = tf.keras.layers.Dense(self.dim_a, activation='linear', name='fc_4')(fc_3_norm)
        policy_output = fc_4

        # Create model
        self.NN_policy = tf.keras.Model(inputs=[observation_input, velocity_input],
                                        outputs=policy_output,
                                        name='policy')
        self.NN_policy.summary()

    def save_policy(self):
        if not os.path.exists(self.network_loc):
            os.makedirs(self.network_loc)
        print("Saving model in: " + self.network_loc + '_policy')
        self.NN_policy.save_weights(self.network_loc + '_policy')

    def _load_policy(self):
        print("Loading model from: " + self.network_loc + '_policy')
        self.NN_policy.load_weights(self.network_loc + '_policy')