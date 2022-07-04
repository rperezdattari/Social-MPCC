import numpy as np
from tools.functions import str_2_array
from buffer import Buffer
import tensorflow as tf

"""
D-COACH implementation
"""


class DCOACH:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, e, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, train_end_episode):
        # Initialize variables
        self.h = None
        self.state_representation = None
        self.policy_action_label = None
        self.e = np.array(str_2_array(e, type_n='float'))
        self.dim_a = dim_a
        self.action_upper_limits = str_2_array(action_upper_limits, type_n='float')
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.train_end_episode = train_end_episode

        # Initialize DCOACH buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    def _generate_policy_label(self, action):
        if np.any(self.h):
            error = np.array(self.h * self.e).reshape(1, self.dim_a)
            self.policy_action_label = []

            for i in range(self.dim_a):
                self.policy_action_label.append(np.clip(action[i] / self.action_upper_limits[i] + error[0, i], -1, 1))

            self.policy_action_label = np.array(self.policy_action_label).reshape(1, self.dim_a)
        else:
            self.policy_action_label = np.reshape(action, [1, self.dim_a])

    @tf.function
    def _update_policy(self, neural_network, state_representation, policy_label):
        # Train policy model
        with tf.GradientTape() as tape_policy:
            policy_output = neural_network.NN_policy([state_representation])
            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - tf.cast(policy_label, dtype=tf.float32)))
            grads = tape_policy.gradient(policy_loss, neural_network.NN_policy.trainable_variables)

        neural_network.policy_optimizer.apply_gradients(zip(grads, neural_network.NN_policy.trainable_variables))

    def _batch_update(self, neural_network, transition_model, batch):
        observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
        action_sequence_batch = [np.array(pair[1]) for pair in batch]
        current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
        action_label_batch = [np.array(pair[3]) for pair in batch]

        batch_size = len(observation_sequence_batch)

        lstm_hidden_state_batch = transition_model.get_lstm_hidden_state_batch(neural_network,
                                                                               observation_sequence_batch,
                                                                               action_sequence_batch,
                                                                               batch_size)

        state_representation_batch = transition_model.get_state_representation_batch(neural_network,
                                                                                     current_observation_batch,
                                                                                     lstm_hidden_state_batch,
                                                                                     batch_size)

        self._update_policy(neural_network, state_representation_batch, action_label_batch)

    def feed_h(self, h, feedback_received):
        self.h = h
        self.feedback_received = feedback_received

    def action(self, neural_network, state_representation):
        self.count += 1
        self.state_representation = state_representation
        action = neural_network.NN_policy(self.state_representation).numpy()
        out_action = []

        for i in range(self.dim_a):
            action[0, i] = np.clip(action[0, i], -1, 1) * self.action_upper_limits[i]
            out_action.append(action[0, i])

        return np.array(out_action)

    def train(self, neural_network, transition_model, action, t, done):
        self._generate_policy_label(action)
        # Policy training
        if self.feedback_received:  # if any element is not 0
            self._update_policy(neural_network, self.state_representation, self.policy_action_label)
            print('agent single update')
            print("feedback:", self.h)

            # Add last step to memory buffer
            if transition_model.last_step(self.policy_action_label) is not None:
                self.buffer.add(transition_model.last_step(self.policy_action_label))

            # Train sampling from buffer
            if self.buffer.initialized():
                print('Train sampling from buffer')
                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)  # TODO: probably this config thing should not be here
                self._batch_update(neural_network, transition_model, batch)

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done):
            print('Train policy every k time steps from buffer')
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            self._batch_update(neural_network, transition_model, batch)