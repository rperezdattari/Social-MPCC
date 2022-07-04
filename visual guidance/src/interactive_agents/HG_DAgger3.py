import numpy as np
from tools.functions import str_2_array
from buffer import Buffer
import tensorflow as tf
import pickle
import time
import cv2
from neural_network3 import NeuralNetwork

"""
Implementation of HG-DAgger without uncertainty estimation
"""


class HG_DAGGER3:
    def __init__(self, dim_a, action_upper_limits, action_lower_limits, buffer_min_size,
                 buffer_max_size, number_training_iterations,  image_width, policy_learning_rate,
                 load_policy, network_loc):
        # Initialize variables
        self.feedback_received = False
        self.image_width = image_width
        self.dim_a = dim_a
        self.action_upper_limits = np.array(str_2_array(action_upper_limits, type_n='float'))
        self.action_lower_limits = np.array(str_2_array(action_lower_limits, type_n='float'))
        self.number_training_iterations = number_training_iterations
        self.last_observation = None
        self.last_velocity_observation = None
        self.last_action = None

        # Initialize Neural Network
        self.neural_network = NeuralNetwork(policy_learning_rate=policy_learning_rate,
                                            load_policy=load_policy,
                                            dim_a=dim_a,
                                            network_loc=network_loc,
                                            image_width=image_width)

        # Initialize HG_DAgger buffer
        self.buffer = Buffer(min_size=buffer_min_size, max_size=buffer_max_size)

    def _preprocess_observation(self, observation):
        observation = cv2.resize(observation, (self.image_width, self.image_width), interpolation=cv2.INTER_AREA) / 255.0
        #self.processed_observation = observation_to_gray(observation, self.image_width)
        self.processed_observation = np.array(observation).reshape([1, self.image_width, self.image_width, 3])
        return tf.convert_to_tensor(self.processed_observation, dtype=tf.float32)

    def _norm_action(self, action):
        action_norm = 2 * (action - (self.action_lower_limits)) / (self.action_upper_limits - self.action_lower_limits) - 1
        return action_norm

    def _denorm_action(self, action_norm):
        action = (action_norm + 1) * (self.action_upper_limits - self.action_lower_limits) / 2 + self.action_lower_limits
        return action

    @tf.function
    def _update_policy(self, features, velocity, policy_label):
        gamma = 0.1
        # Train policy model
        with tf.GradientTape() as tape_policy:
            policy_output = self._eval_action(features, velocity)
            velocity_map = tf.clip_by_value(((velocity + 1) * 5), clip_value_min=0, clip_value_max=8) / 4 - 1
            policy_loss = tf.reduce_mean(tf.square(policy_output - tf.cast(policy_label, dtype=tf.float32)))\
                          + gamma * tf.reduce_mean(tf.square(policy_output - tf.cast(velocity_map, dtype=tf.float32)))
            grads = tape_policy.gradient(policy_loss, self.neural_network.NN_policy.trainable_variables)

        self.neural_network.policy_optimizer.apply_gradients(zip(grads, self.neural_network.NN_policy.trainable_variables))

    @tf.function
    def _eval_action(self, observation_image, observation_velocity):
        action = self.neural_network.NN_policy([observation_image, observation_velocity])
        return tf.clip_by_value(action, clip_value_min=self.action_lower_limits, clip_value_max=self.action_upper_limits)

    def _errors_dataset(self, dataset, batch_size):
        size_dataset = len(dataset)
        rest = size_dataset % batch_size
        errors = []

        if rest != 0:
            iterations = size_dataset // batch_size + 1
        else:
            iterations = size_dataset // batch_size

        for i in range(iterations):
            if rest != 0 and (i == iterations - 1):
                batch_size = rest
                batch = dataset[-batch_size:]
            else:
                batch = dataset[i * batch_size:(i + 1) * batch_size]

            features_batch = np.reshape([np.array(pair[0]) for pair in batch], [batch_size,
                                                                                self.image_width,
                                                                                self.image_width,
                                                                                3])
            velocity_batch = np.reshape([np.array(pair[1]) for pair in batch], [batch_size, 1])
            action_label_batch = np.reshape([np.array(pair[2]) for pair in batch], [batch_size, self.dim_a])

            action = self._eval_action(features_batch, velocity_batch)

            errors_batch = tf.square(action - action_label_batch)
            errors.append(errors_batch)

        if rest == 0:
            errors_tensor = tf.stack(errors)
            errors_tensor_out = tf.reshape(errors_tensor, [errors_tensor.shape[0] * errors_tensor.shape[1], -1])
        else:
            errors_tensor = tf.stack(errors[:-1])
            errors_tensor_out_partial = tf.reshape(errors_tensor, [errors_tensor.shape[0] * errors_tensor.shape[1], -1])
            errors_tensor_out = tf.concat([errors_tensor_out_partial, errors[-1]], axis=0)

        return errors_tensor_out

    # Sampling strategy 1
    def _get_sampling_probability(self, errors):
        sum_errors = tf.math.reduce_sum(errors, axis=0)
        sampling_probability_tensor = errors / sum_errors
        sampling_probability = sampling_probability_tensor.numpy()[:, 0]
        return sampling_probability

    def _sample_from_probability(self, batch_size, sampling_probability):
        dataset_size = len(self.buffer.buffer)
        sampled_indexes = np.random.choice(np.arange(dataset_size), size=batch_size, p=sampling_probability)
        batch = np.array(self.buffer.buffer)[sampled_indexes]
        return batch

    # Sampling strategy 2
    def _split_dataset(self, errors, batch_size, alpha, min_buffer_out_std_size, buffer_out_std_sampling_probability):
        errors = errors.numpy()[:, 0]
        # Get statistics errors
        errors_mean = np.mean(errors)
        errors_std = np.std(errors)

        # Compute gaussian thresholds
        upper_threshold = errors_mean + alpha * errors_std
        lower_threshold = errors_mean - alpha * errors_std

        # Find values inside thresholds
        errors_upper_index = errors > lower_threshold
        errors_lower_index = errors < upper_threshold

        # Create buffer with values inside std
        samples_in_std = np.array(self.buffer.buffer)[errors_upper_index & errors_lower_index]
        buffer_in_std = Buffer(min_size=0, max_size=1e16)
        buffer_in_std.buffer = list(samples_in_std)
        batch_size_1 = int(batch_size * (1 - buffer_out_std_sampling_probability))

        # Crate buffer with values outside std
        samples_out_std = np.array(self.buffer.buffer)[~(errors_upper_index & errors_lower_index)]
        buffer_out_std = Buffer(min_size=min_buffer_out_std_size, max_size=1e16)
        buffer_out_std.buffer = list(samples_out_std)
        batch_size_2 = int(batch_size * buffer_out_std_sampling_probability)

        return buffer_in_std, buffer_out_std, batch_size_1, batch_size_2

    def _sample_from_split(self, buffer_in_std, buffer_out_std, batch_size, batch_size_1, batch_size_2):
        if not buffer_out_std.initialized():
            batch = self.buffer.sample(batch_size=batch_size)
        else:
            # Sample from in std buffer
            batch_1 = buffer_in_std.sample(batch_size=batch_size_1)

            # Sample from out std buffer
            batch_2 = buffer_out_std.sample(batch_size=batch_size_2)

            # Combine both batches
            batch = batch_1 + batch_2

        return batch

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

    def action(self, observation):
        image_observation = observation[0]
        velocity_observation = tf.reshape(observation[1], [1, 1])
        self.last_velocity_observation = velocity_observation

        # Get features from pretrained networks
        image_observation = self._preprocess_observation(np.array(image_observation))
        self.last_observation = image_observation[0]

        # Get action human
        action_human = self.h[0]

        # Get action agent
        action_agent = self._eval_action(image_observation, velocity_observation)[0].numpy()

        if self.feedback_received:  # if feedback, human teleoperates
            action = action_human
            print('feedback:', action)

            # Normalize action to store in buffer
            action_norm = self._norm_action(action)

            # Add state-action to buffer
            self.buffer.add([tf.reshape(image_observation, -1), velocity_observation, action_norm])
        else:
            action = action_agent

            # Denorm action
            action = self._denorm_action(action)

            print('action:', action)

        self.last_action = action
        return action, action_human, action_agent

    def train(self, batch_size, feedback_received, done):
        probability_update_frequency = self.number_training_iterations // 10

        if self.buffer.initialized() and done:
            for i in range(self.number_training_iterations):
                if i % (self.number_training_iterations / 20) == 0:
                    print('Progress Policy training: %i %%' % (i / self.number_training_iterations * 100))

                if i < probability_update_frequency:
                    batch = self.buffer.sample(batch_size=batch_size)
                else:
                    if (i % probability_update_frequency) == 0:
                        errors = self._errors_dataset(dataset=self.buffer.buffer, batch_size=200)
                        print('Errors updated!')
                        buffer_in_std, buffer_out_std, batch_size_1, batch_size_2 = self._split_dataset(errors,
                                                                                                        batch_size=batch_size,
                                                                                                        alpha=1,
                                                                                                        min_buffer_out_std_size=30,
                                                                                                        buffer_out_std_sampling_probability=0.4)
                        print('Size buffer 1:', len(buffer_in_std.buffer), '; Size buffer 2:', len(buffer_out_std.buffer))
                        # sampling_probability = self._get_sampling_probability(errors)
                    batch = self._sample_from_split(buffer_in_std, buffer_out_std, batch_size, batch_size_1, batch_size_2)
                    #batch = self._sample_from_probability(batch_size, sampling_probability)

                # Sample demonstrations from buffer and reshape
                batch_size_current = len(batch)
                observation_batch = np.reshape([np.array(pair[0]) for pair in batch], [batch_size_current,
                                                                                       self.image_width,
                                                                                       self.image_width,
                                                                                       3])
                velocity_batch = np.reshape([np.array(pair[1]) for pair in batch], [batch_size_current, 1])
                action_label_batch = np.reshape([np.array(pair[2]) for pair in batch], [batch_size_current, self.dim_a])

                # Train policy
                self._update_policy(observation_batch,
                                    velocity_batch,
                                    action_label_batch)

        if self.buffer.initialized() and feedback_received:
            # Sample demonstrations from buffer and reshape
            batch = self.buffer.sample(batch_size=batch_size)
            observation_batch = np.reshape([np.array(pair[0]) for pair in batch], [batch_size,
                                                                                   self.image_width,
                                                                                   self.image_width,
                                                                                   3])
            velocity_batch = np.reshape([np.array(pair[1]) for pair in batch], [batch_size, 1])
            action_label_batch = np.reshape([np.array(pair[2]) for pair in batch], [batch_size, self.dim_a])

            # Replace first element with current feedback
            observation_batch[0] = self.last_observation
            velocity_batch[0] = self.last_velocity_observation
            action_label_batch[0] = self.last_action

            # Train policy
            self._update_policy(observation_batch,
                                velocity_batch,
                                action_label_batch)