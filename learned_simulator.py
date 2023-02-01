import graph_nets as gn
import sonnet as snt
import tensorflow as tf

import connectivity_utils
import graph_network


class LearnedSimulator(snt.Module):
    def __init__(
            self,
            num_dimensions,
            connectivity_radius,
            boundaries,
            num_boundaries,
            num_particle_types,
            num_processing_steps,
            particle_type_embedding_size,
            name="LearnedSimulator"):

        super().__init__(name=name)

        self._connectivity_radius = connectivity_radius
        self._num_particle_types = num_particle_types
        self._num_processing_steps = num_processing_steps
        self._boundaries = boundaries
        self.num_boundaries = num_boundaries
        self.graph_networks = graph_network.GraphEPD(node_output_size=num_dimensions)

        if self._num_particle_types > 1:
            self._particle_type_embedding = tf.Variable(
                tf.ones(shape=[self._num_particle_types, particle_type_embedding_size]),
                trainable=True,
                name="particle_embedding"
            )

    def get_predicted(
            self,
            position_sequence,
            n_particles_per_example,
            particle_types=None
    ):

        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence,
            n_particles_per_example,
            particle_types
        )

        predicted_velocity = self.graph_networks(input_graphs_tuple, self._num_processing_steps)
        next_position = self._decoder_postprocessor(normalized_velocity[-1].nodes, position_sequence)
        return next_position

    def _encoder_preprocessor(
            self,
            position_sequence,
            n_node,
            particle_types
    ):

        most_recent_position = position_sequence[:, -1]
        input_position = position_sequence[:, :1]
        boundary_sequence, n_bound = self._boundary_sequence(input_position[0, 0, -1])
        position_boundary = tf.concat([input_position, boundary_sequence], axis=0)
        n_nodes = n_node + n_bound

        (senders, receivers, n_edge) = connectivity_utils.compute_connectivity_for_batch_pyfunc(
            position_boundary, n_nodes, self._connectivity_radius
        )

        node_features = []
        flat_position_sequence = snt.reshape(input_position, output_shape=(-1,))
        node_features.append(flat_position_sequence)

        # todo: boundary sequence 에 particle type 부여 할 것
        ## particle_types: (678,)
        ## self._particle_type_embedding: (9, 16)
        if self._num_particle_types > 1:
            particle_type_embeddings = tf.nn.embedding_lookup(self._particle_type_embedding, particle_types)
            node_features.append(particle_type_embeddings)

        edge_features = []

        normalized_relative_displacements = (tf.gather(most_recent_position, senders) -
                                             tf.gather(most_recent_position, receivers)) / self._connectivity_radius
        edge_features.append(normalized_relative_displacements)

        normalized_relative_distances = tf.norm(normalized_relative_displacements, axis=-1, keepdims=True)
        edge_features.append(normalized_relative_distances)

        global_context = tf.zeros((tf.shape(n_nodes)[0], 1))

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
            position_sequence,
            n_particles_per_example,
            particle_types=None
    ):

        input_graphs_tuple = self._encoder_preprocessor(
            position_sequence,
            n_particles_per_example,
            particle_types
        )
        predicted_velocity = self.graph_networks(input_graphs_tuple, self._num_processing_steps)
        target_velocity = time_diff(position_sequence)

        return predicted_velocity, target_velocity

    def _inverse_decoder_postprocessor(
            self,
            next_position,
            position_sequence):

        previous_position = position_sequence[:, -1]
        velocity = next_position - previous_position
        return velocity

    def _boundary_sequence(self, times):
        const = tf.constant(self._boundaries, dtype=tf.float32)
        times = tf.linspace(times, times, self.num_boundaries)[:, tf.newaxis, tf.newaxis]
        x_min_min = tf.linspace(const[0, 0], const[0, 0], self.num_boundaries)[:, tf.newaxis, tf.newaxis]
        x_max_max = tf.linspace(const[0, 1], const[0, 1], self.num_boundaries)[:, tf.newaxis, tf.newaxis]
        x_min_max = tf.linspace(const[0, 0], const[0, 1], self.num_boundaries)[:, tf.newaxis, tf.newaxis]
        y_min_min = tf.linspace(const[1, 0], const[1, 0], self.num_boundaries)[:, tf.newaxis, tf.newaxis]
        y_max_max = tf.linspace(const[1, 1], const[1, 1], self.num_boundaries)[:, tf.newaxis, tf.newaxis]
        y_min_max = tf.linspace(const[1, 0], const[1, 1], self.num_boundaries)[:, tf.newaxis, tf.newaxis]

        boundaries = tf.concat([
            tf.concat([x_min_max, y_min_min, times], axis=2),
            tf.concat([x_min_max, y_max_max, times], axis=2),
            tf.concat([x_min_min, y_min_max, times], axis=2),
            tf.concat([x_max_max, y_min_max, times], axis=2)
        ], axis=0)

        n_boundary = boundaries.get_shape().as_list()[0]
        return boundaries, n_boundary


def time_diff(input_sequence):
    distance = input_sequence[:, 1:, :-1] - input_sequence[:, :-1, :-1]
    spend_time = input_sequence[:, 1:, -1] - input_sequence[:, :-1, -1]
    return distance / tf.expand_dims(spend_time, axis=1)
