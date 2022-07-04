import numpy as np
from tools.functions import observation_to_gray, FastImagePlot
from buffer import Buffer
import cv2
import tensorflow as tf
"""
Transition model
"""


class TransitionModel:
    def __init__(self, training_sequence_length, lstm_hidden_state_size, crop_observation, image_width,
                 show_transition_model_output, show_observation, resize_observation, occlude_observation, dim_a,
                 buffer_sampling_rate, buffer_sampling_size, number_training_iterations, train_end_episode):

        self.lstm_h_size = lstm_hidden_state_size
        self.dim_a = dim_a
        self.training_sequence_length = training_sequence_length
        self.number_training_iterations = number_training_iterations
        self.train_end_episode = train_end_episode

        # System model parameters
        self.lstm_hidden_state = [tf.zeros([1, self.lstm_h_size]), tf.zeros([1, self.lstm_h_size])]
        self.image_width = image_width  # we assume that images are squares

        # High-dimensional observation initialization
        self.resize_observation = resize_observation
        self.show_observation = show_observation
        self.show_ae_output = show_transition_model_output
        self.t_counter = 0
        self.crop_observation = crop_observation
        self.occlude_observation = occlude_observation

        # Buffers
        self.last_actions = Buffer(min_size=self.training_sequence_length + 1,
                                   max_size=self.training_sequence_length + 1)
        self.last_actions.add(np.zeros([1, self.dim_a]))
        self.last_states = Buffer(min_size=self.training_sequence_length + 1,
                                  max_size=self.training_sequence_length + 1)
        self.last_states.add(np.zeros([1, self.image_width, self.image_width, 3]))
        self.transition_model_buffer_sampling_rate = buffer_sampling_rate
        self.transition_model_sampling_size = buffer_sampling_size

        if self.show_observation:
            self.state_plot = FastImagePlot(1, np.zeros([self.image_width, self.image_width, 3]),
                                            self.image_width, 'Image State', vmax=255)

    def _preprocess_observation(self, observation):
        if self.resize_observation:
            observation = cv2.resize(observation, (self.image_width, self.image_width), interpolation=cv2.INTER_AREA)

        #self.processed_observation = observation_to_gray(observation, self.image_width)
        self.processed_observation = np.array(observation).reshape([1, self.image_width, self.image_width, 3])
        self.last_states.add(self.processed_observation)
        self.network_input = np.array(self.last_states.buffer)

        self.network_input = tf.convert_to_tensor(self.network_input, dtype=tf.float32)

    def _refresh_image_plots(self):
        if self.t_counter % 4 == 0 and self.show_observation:
            self.state_plot.refresh(self.processed_observation)

    @tf.function
    def _extract_features(self, neural_network, NN_input):
        features = neural_network.NN_feature_extraction(NN_input)
        return features

    def get_state_representation(self, neural_network, observation):
        self._preprocess_observation(np.array(observation))

        reshaped_network_input = tf.reshape(self.network_input[-1], [1, self.image_width, self.image_width, 3])

        NN_input = reshaped_network_input

        features = self._extract_features(neural_network, NN_input)

        self._refresh_image_plots()  # refresh image plots
        self.t_counter += 1

        return features

    def get_state_representation_batch(self, neural_network, current_observation, batch_size):
        feature_extraction_input = tf.convert_to_tensor(np.reshape(current_observation, [batch_size,
                                                                                         self.image_width,
                                                                                         self.image_width,
                                                                                         3]),
                                                        dtype=tf.float32)

        NN_input = feature_extraction_input

        features = self._extract_features(neural_network, NN_input)

        return features

    def last_step(self, action_label):
        if self.last_states.initialized() and self.last_actions.initialized():
            return [self.network_input[:-1],
                    self.last_actions.buffer[:-1],
                    self.network_input[-1],
                    action_label.reshape(self.dim_a)]
        else:
            return None

    def add_action(self, action):
        self.last_actions.add(action)

    def new_episode(self):
        self.lstm_hidden_state = [tf.zeros([1, self.lstm_h_size]), tf.zeros([1, self.lstm_h_size])]
        self.last_states = Buffer(min_size=self.training_sequence_length + 1,
                                  max_size=self.training_sequence_length + 1)
        self.last_actions = Buffer(min_size=self.training_sequence_length + 1,
                                   max_size=self.training_sequence_length + 1)
        self.last_actions.add(np.zeros([1, self.dim_a]))
        self.last_states.add(np.zeros([1, self.image_width, self.image_width, 3]))