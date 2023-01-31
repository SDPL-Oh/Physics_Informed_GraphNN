import graph_nets as gn
import sonnet as snt
import tensorflow as tf

import connectivity_utils
import graph_network

STD_EPSILON = 1e-8


class LearnedSimulator(snt.Module):
    def __init__(
            self,
            num_dimensions,
            connectivity_radius,
            boundaries,
            num_particle_types,
            num_processing_steps,
            particle_type_embedding_size,
            name="LearnedSimulator"):

        super().__init__(name=name)

        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._num_processing_steps = num_processing_steps
        self._boundaries = boundaries
        self.graph_networks = graph_network.GraphEPD(node_output_size=num_dimensions)

        if self._num_particle_types > 1:
            self._particle_type_embedding = tf.Variable(
                tf.ones(shape=[self._num_particle_types, particle_type_embedding_size]),
                trainable=True,
                name="particle_embedding"
            )

    def build(
            self,
            position_sequence,
            n_particles_per_example,
            n_boundary_per_example,
            particle_types=None
    ):

        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence, n_particles_per_example, n_boundary_per_example, particle_types
        )
        normalized_velocity = self.graph_networks(input_graphs_tuple, self._num_processing_steps)
        next_position = self._decoder_postprocessor(normalized_velocity[-1].nodes, position_sequence)
        return next_position

    def _encoder_preprocessor(
            self,
            position_sequence,
            n_node,
            n_boundary,
            particle_types
    ):

        most_recent_position = position_sequence[:, -1]
        velocity_sequence = time_diff(position_sequence)


        boundaries = tf.constant(self._boundaries, dtype=tf.float32)
        distance_to_lower_boundary = (most_recent_position - tf.expand_dims(boundaries[:, 0], 0))
        distance_to_upper_boundary = (tf.expand_dims(boundaries[:, 1], 0) - most_recent_position)
        distance_to_boundaries = tf.concat([distance_to_lower_boundary, distance_to_upper_boundary], axis=1)

        normalized_clipped_distance_to_boundaries = tf.clip_by_value(
            distance_to_boundaries / self._connectivity_radius, -1., 1.
        )

        print(normalized_clipped_distance_to_boundaries)

        # position_boundary = tf.concat([most_recent_position, most_recent_boundary], axis=0)
        n_node = n_node + n_boundary

        (senders, receivers, n_edge) = connectivity_utils.compute_connectivity_for_batch_pyfunc(
            most_recent_position, n_node, self._connectivity_radius
        )

        node_features = []
        flat_position_sequence = snt.reshape(position_sequence, output_shape=(-1,))
        node_features.append(flat_position_sequence)
        node_features.append(time_sequence)

        boundaries = tf.constant(self._boundaries, dtype=tf.float32)
        distance_to_lower_boundary = (most_recent_position - tf.expand_dims(boundaries[:, 0], 0))
        distance_to_upper_boundary = (tf.expand_dims(boundaries[:, 1], 0) - most_recent_position)
        distance_to_boundaries = tf.concat([distance_to_lower_boundary, distance_to_upper_boundary], axis=1)
        normalized_clipped_distance_to_boundaries = tf.clip_by_value(
            distance_to_boundaries / self._connectivity_radius, -1., 1.
        )

        node_features.append(normalized_clipped_distance_to_boundaries)

        if self._num_particle_types > 1:
            particle_type_embeddings = tf.nn.embedding_lookup(self._particle_type_embedding, particle_types)
            node_features.append(particle_type_embeddings)
        edge_features = []

        normalized_relative_displacements = (tf.gather(most_recent_position, senders) -
                                             tf.gather(most_recent_position, receivers)) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = tf.norm(normalized_relative_displacements, axis=-1, keepdims=True)
        edge_features.append(normalized_relative_distances)

        global_context = tf.zeros((tf.shape(n_node)[0], 1))

        return gn.graphs.GraphsTuple(
            nodes=tf.concat(node_features, axis=-1),
            edges=tf.concat(edge_features, axis=-1),
            globals=global_context,
            n_node=n_node,
            n_edge=n_edge,
            senders=senders,
            receivers=receivers,
        )

    def _decoder_postprocessor(
            self,
            velocity,
            position_sequence):

        most_recent_position = position_sequence[:, -1]
        new_position = most_recent_position + velocity  # * dt = 1
        return new_position

    def get_predicted_and_target_velocity(
            self,
            position_sequence_noise,
            position_sequence,
            n_particles_per_example,
            particle_types=None
    ):

        noisy_position_sequence = position_sequence + position_sequence_noise

        n_boundary = 1

        input_graphs_tuple = self._encoder_preprocessor(
            noisy_position_sequence,
            n_particles_per_example,
            n_boundary,
            particle_types
        )
        predicted_velocity = self.graph_networks(input_graphs_tuple, self._num_processing_steps)

        next_position_adjusted = next_position + position_sequence_noise[:, -1]
        target_velocity = self._inverse_decoder_postprocessor(
            next_position_adjusted,
            noisy_position_sequence
        )

        return predicted_velocity, target_velocity

    def _inverse_decoder_postprocessor(
            self,
            next_position,
            position_sequence):

        previous_position = position_sequence[:, -1]
        velocity = next_position - previous_position
        return velocity


def time_diff(input_sequence):
    distance = input_sequence[:, 1:, :-1] - input_sequence[:, :-1, :-1]
    spend_time = input_sequence[:, 1:, -1] - input_sequence[:, :-1, -1]
    return distance / tf.expand_dims(spend_time, axis=1)
