import numpy as np
from tools.functions import str_2_array
from buffer import Buffer
import tensorflow as tf
import pickle
import time

"""
Implementation of HG-DAgger without uncertainty estimation
"""


class HG_DAGGER:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, buffer_min_size, buffer_max_size,
                 buffer_sampling_rate, buffer_sampling_size, number_training_iterations, train_end_episode,
                 lstm_hidden_state_size):
        # Initialize variables
        self.feature_extractor = 'vgg'  # Options: vgg, resnet

        if self.feature_extractor == 'vgg':
            self.features_dim = 512
        elif self.feature_extractor == 'resnet':
            self.features_dim = 2048

        self.sequence_length = 10
        self.image_width = 64
        self.batch_size = 20
        self.dim_a = dim_a
        self.action_upper_limits = str_2_array(action_upper_limits, type_n='float')
        self.action_lower_limits = str_2_array(action_lower_limits, type_n='float')
        self.count = 0
        self.buffer_sampling_rate = buffer_sampling_rate
        self.buffer_sampling_size = buffer_sampling_size
        self.number_training_iterations = number_training_iterations
        self.train_end_episode = train_end_episode
        self.lstm_h_size = 150
        self.lstm_zero_hidden_state_batch = [tf.zeros([self.batch_size, self.lstm_h_size]), tf.zeros([self.batch_size, self.lstm_h_size])]
        self.lstm_hidden_state = [tf.zeros([1, self.lstm_h_size]), tf.zeros([1, self.lstm_h_size])]

        # Initialize HG_DAgger buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    @tf.function
    def _update_policy(self, neural_network, state_representation, policy_label):
        # Train policy model
        with tf.GradientTape() as tape_policy:
            policy_output = self._eval_action(neural_network,
                                              state_representation,
                                              batch_size=self.batch_size,
                                              sequence_length=self.sequence_length,
                                              lstm_hidden_state=self.lstm_zero_hidden_state_batch)[0]
            policy_loss = 0.5 * tf.reduce_mean(tf.square(policy_output - tf.cast(policy_label, dtype=tf.float32)))
            grads = tape_policy.gradient(policy_loss, neural_network.NN_policy.trainable_variables)

        neural_network.policy_optimizer.apply_gradients(zip(grads, neural_network.NN_policy.trainable_variables))
        return policy_loss

    @tf.function
    def _eval_action(self, neural_network, state_representation, batch_size, sequence_length, lstm_hidden_state):
        action = neural_network.NN_policy([tf.reshape(state_representation, [batch_size, sequence_length, self.features_dim]),
                                           lstm_hidden_state])
        return action

    def save_buffer(self):
        time_stamp = str(int(time.time() * 100))[-10:]
        file_handler = open('online_data/HG_DAgger/HG_DAgger_buffer_%s.obj' % time_stamp, 'wb')
        pickle.dump(self.buffer, file_handler)

    def load_buffer(self, dir):
        file_handler = open(dir, 'rb')
        self.buffer = pickle.load(file_handler)

    def feed_h(self, h, feedback_received):
        self.h = np.reshape(h, [1, self.dim_a])
        self.feedback_received = feedback_received

    def action(self, neural_network, state_representation):
        self.count += 1

        action_human = self.h
        action_agent, self.lstm_hidden_state = self._eval_action(neural_network,
                                                                 state_representation=state_representation,
                                                                 batch_size=1,
                                                                 sequence_length=1,
                                                                 lstm_hidden_state=self.lstm_hidden_state)

        if self.feedback_received:  # if feedback, human teleoperates
            action = action_human
            print("feedback:", self.h[0])
        else:
            action = np.clip(action_agent.numpy(),-1,1)

        return action, action_human, np.clip(action_agent,-1,1)

    def last_step(self, action_label):
        if self.last_states.initialized() and self.last_actions.initialized():
            return [self.network_input[:-1],
                    self.last_actions.buffer[:-1],
                    self.network_input[-1],
                    action_label.reshape(self.dim_a)]
        else:
            return None

    def train(self, neural_network, transition_model, action, t, done, summary_writer, t_train, feedback_received, last_step):
        # Add last step to memory buffer
        if transition_model.last_step(action) is not None and self.feedback_received:  # if human teleoperates, add action to database
            self.buffer.add(transition_model.last_step(action))

        # Train policy every k time steps from buffer
        if self.buffer.initialized() and (t % self.buffer_sampling_rate == 0 or (self.train_end_episode and done)):
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))

                batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
                observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
                current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
                action_label_batch = [np.array(pair[3]) for pair in batch]
                batch_size = len(current_observation_batch)  # The zero is here because no lstm

                observation_sequence_batch = np.reshape(observation_sequence_batch, [batch_size * self.sequence_length, self.image_width, self.image_width, 3])

                # Compute state representation
                state_representation_batch = transition_model.get_state_representation_batch(neural_network,
                                                                                             observation_sequence_batch,
                                                                                             batch_size * self.sequence_length)

                loss = self._update_policy(neural_network,
                                           tf.reshape(state_representation_batch,
                                                      [batch_size, self.sequence_length, self.features_dim]),
                                           np.reshape(action_label_batch, [self.batch_size, self.dim_a]))
                #with summary_writer.as_default():
                #    tf.summary.scalar('loss', loss, step=t_train)
                #t_train += 1

        if False:#self.buffer.initialized() and feedback_received:
            batch = self.buffer.sample(batch_size=self.buffer_sampling_size)
            observation_sequence_batch = [np.array(pair[0]) for pair in batch]  # state(t) sequence
            current_observation_batch = [np.array(pair[2]) for pair in batch]  # last
            action_label_batch = [np.array(pair[3]) for pair in batch]
            batch_size = len(current_observation_batch)  # The zero is here because no lstm

            # Replace first element with current feedback
            observation_sequence_batch[0] = last_step[0]
            current_observation_batch[0] = last_step[2]
            action_label_batch[0] = last_step[3]

            # Compute state representation
            state_representation_batch = transition_model.get_state_representation_batch(neural_network,
                                                                                         observation_sequence_batch,
                                                                                         batch_size * self.sequence_length)

            loss = self._update_policy(neural_network,
                                       tf.reshape(state_representation_batch,
                                                  [batch_size, self.sequence_length, self.features_dim]),
                                       np.reshape(action_label_batch, [self.batch_size, self.dim_a]))
            #with summary_writer.as_default():
            #    tf.summary.scalar('loss', loss, step=t_train)
            #t_train += 1
        return t_train

    def new_episode(self):
        self.lstm_hidden_state = [tf.zeros([1, self.lstm_h_size]), tf.zeros([1, self.lstm_h_size])]