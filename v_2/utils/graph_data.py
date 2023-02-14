import graph_nets as gn
import sonnet as snt
import tensorflow as tf
import numpy as np
from sklearn import neighbors

import connectivity_utils
import graph_network

import learned_simulator


def get_random_walk_noise_for_position_sequence(position_sequence, noise_std_last_step):
    velocity_sequence = learned_simulator.time_diff(position_sequence)

    num_velocities = velocity_sequence.shape.as_list()[1]
    velocity_sequence_noise = tf.random.normal(
        tf.shape(velocity_sequence),
        stddev=noise_std_last_step / num_velocities ** 0.5,
        dtype=position_sequence.dtype)

    velocity_sequence_noise = tf.cumsum(velocity_sequence_noise, axis=1)

    position_sequence_noise = tf.concat([
        tf.zeros_like(velocity_sequence_noise[:, 0:1]),
        tf.cumsum(velocity_sequence_noise, axis=1)], axis=1)

    return position_sequence_noise


def time_diff(input_sequence):
    distance = input_sequence[:, 1:, :-1] - input_sequence[:, :-1, :-1]
    spend_time = input_sequence[:, 1:, -1] - input_sequence[:, :-1, -1]
    return distance / tf.expand_dims(spend_time, axis=1)


def boundary_sequence(boundaries, num_boundaries, times):
    const = tf.constant(boundaries, dtype=tf.float32)
    times = tf.linspace(times, times, num_boundaries)[:, tf.newaxis]
    x_min_min = tf.linspace(const[0, 0], const[0, 0], num_boundaries)[:, tf.newaxis]
    x_max_max = tf.linspace(const[0, 1], const[0, 1], num_boundaries)[:, tf.newaxis]
    x_min_max = tf.linspace(const[0, 0], const[0, 1], num_boundaries)[:, tf.newaxis]
    y_min_min = tf.linspace(const[1, 0], const[1, 0], num_boundaries)[:, tf.newaxis]
    y_max_max = tf.linspace(const[1, 1], const[1, 1], num_boundaries)[:, tf.newaxis]
    y_min_max = tf.linspace(const[1, 0], const[1, 1], num_boundaries)[:, tf.newaxis]
    boundaries = tf.concat([
        tf.concat([x_min_max, y_min_min, times], axis=1),
        tf.concat([x_min_max, y_max_max, times], axis=1),
        tf.concat([x_min_min, y_min_max, times], axis=1),
        tf.concat([x_max_max, y_min_max, times], axis=1)
    ], axis=0)

    n_boundary = boundaries.get_shape().as_list()[0]
    return boundaries, n_boundary


def boundary_velocity(self):
    zeros = tf.zeros([self.num_boundaries * 4, 2])
    return zeros


def decoder_postprocessor(velocity, position_sequence):
    most_recent_position = position_sequence[:, -1]
    new_position = most_recent_position + velocity
    return new_position


def compute_connectivity(positions, radius, add_self_edges):
    tree = neighbors.KDTree(positions)
    receivers_list = tree.query_radius(positions, r=radius)
    num_nodes = len(positions)
    senders = np.repeat(range(num_nodes), [len(a) for a in receivers_list])
    receivers = np.concatenate(receivers_list, axis=0)

    if not add_self_edges:
        mask = senders != receivers
        senders = senders[mask]
        receivers = receivers[mask]

    return senders, receivers


def compute_connectivity_for_batch(positions, n_node, radius, add_self_edges):
    positions_per_graph_list = np.split(positions, np.cumsum(n_node[:-1]), axis=0)
    receivers_list = []
    senders_list = []
    n_edge_list = []
    num_nodes_in_previous_graphs = 0

    for positions_graph_i in positions_per_graph_list:
        senders_graph_i, receivers_graph_i = compute_connectivity(
            positions_graph_i, radius, add_self_edges)

        num_edges_graph_i = len(senders_graph_i)
        n_edge_list.append(num_edges_graph_i)

        receivers_list.append(receivers_graph_i + num_nodes_in_previous_graphs)
        senders_list.append(senders_graph_i + num_nodes_in_previous_graphs)

        num_nodes_graph_i = len(positions_graph_i)
        num_nodes_in_previous_graphs += num_nodes_graph_i

    senders = np.concatenate(senders_list, axis=0).astype(np.int32)
    receivers = np.concatenate(receivers_list, axis=0).astype(np.int32)
    n_edge = np.stack(n_edge_list).astype(np.int32)

    return senders, receivers, n_edge


def compute_connectivity_for_batch_pyfunc(positions, n_node, radius, add_self_edges=True):
    senders, receivers, n_edge = compute_connectivity_for_batch(positions, n_node, radius, add_self_edges)
    return senders, receivers, n_edge


class HouseGan:
    def __init__(self, FLAGS):
        self.input_size = FLAGS.input_size
        self.output_size = FLAGS.output_size
        self.num_class = hparams['num_class']
        self.latent = hparams['latent']
        self.batch = hparams['batch']
        self.img_size = hparams['img_size']
        self.num_variations = hparams['num_variations']
        self.epochs = hparams['epochs']
        self.generator_lr = hparams['generator_lr']
        self.discriminator_lr = hparams['discriminator_lr']
        self.num_process = hparams['num_process']
        self.decay_steps = hparams['decay_steps']
        self.decay_rate = hparams['decay_rate']
        self.model_path = hparams['model_path']
        self.plt_path = hparams['plt_path']
        self.log_path = hparams['log_path']
        self.train_data = hparams['train_data']
        self.test_data = hparams['test_data']

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.generator_lr,
            decay_steps=self.decay_steps,
            decay_rate=self.decay_rate,
            staircase=True)
        self.generator = Generator(node_output_size=2)
        self.discriminator = Discriminator(node_output_size=1)
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)
        self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=self.discriminator_lr)

    def base_graph(self, node):
        nodes = np.zeros([node, 3], np.float32)
        senders, receivers, edges = compute_connectivity_for_batch(positions, n_node, radius, add_self_edges)
        return {
            "globals": [0.],
            "nodes": nodes,
            "edges": edges,
            "receivers": receivers,
            "senders": senders
        }

    def graphTuple(self, nodes):
        batches_graph = []
        for node in nodes:
            init = self.base_graph(len(node))
            batches_graph.append(init)
        input_tuple = utils_tf.data_dicts_to_graphs_tuple(batches_graph)
        return input_tuple

    def generateGraph(self, input_op, output_ops):
        generate_ops = [
            tf.concat([output_op.nodes, input_op.nodes[:, :12]], axis=-1) for output_op in output_ops]
        return tf.stack(generate_ops)

    def oneSingleLoss(self, target_ops, output_ops):
        loss_ops = [self.cross_entropy(tf.ones_like(target_op.nodes), output_op.nodes)
                    for target_op, output_op in zip(target_ops, output_ops)]
        return tf.stack(loss_ops)

    def zeroSingleLoss(self, target_ops, output_ops):
        loss_ops = [self.cross_entropy(tf.zeros_like(target_op.nodes), output_op.nodes)
                    for target_op, output_op in zip(target_ops, output_ops)]
        return tf.stack(loss_ops)

    def averageLoss(self, lbl, prd, one_zero='one'):
        if one_zero == 'zero':
            per_example_loss = self.zeroSingleLoss(lbl, prd)
        else:
            per_example_loss = self.oneSingleLoss(lbl, prd)
        return tf.math.reduce_sum(per_example_loss) / self.num_process

    def initSpec(self):
        init = utils_tf.data_dicts_to_graphs_tuple([self.baseGraphsNp])
        return utils_tf.specs_from_graphs_tuple(init)

    def generatorLoss(self, fake_output):
        return self.averageLoss(fake_output, fake_output, 'one')

    def discriminatorLoss(self, real_output, fake_output):
        real_loss = self.averageLoss(real_output, real_output, 'one')
        fake_loss = self.averageLoss(fake_output, fake_output, 'zero')
        total_loss = real_loss + fake_loss
        return total_loss

    @tf.function
    def train_step(self, x, y):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_output = self.generator(x, self.num_process)
            generated_graph = self.generateGraph(x, generated_output)

            output_tuple = self.graphTuple(generated_graph, 'outputs')
            output_tuple_tf = output_tuple.replace(nodes=generated_graph[0])

            real_output = self.discriminator(y, 1)
            fake_output = self.discriminator(output_tuple_tf, 1)

            gen_loss = self.generatorLoss(fake_output)
            disc_loss = self.discriminatorLoss(real_output, fake_output)

        gradients_of_generator = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

        self.generator_opt.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        return gen_loss, disc_loss

    @tf.function
    def predict_step(self, x):
        return self.generator(x, self.num_process)

    def plotStep(self, inputs, outputs, height, width, filename):
        loss_ops = [self.spacePlot(per_input, output.nodes, height, width, filename)
                    for per_input, output in zip(tf.split(inputs.nodes, self.batch, axis=1), outputs)]
        return loss_ops

    def plotTarget(self, inputs, outputs, height, width, filename):
        loss_ops = [self.spacePlot(per_input, output, height, width, filename)
                    for per_input, output in zip(tf.split(inputs.nodes, self.batch, axis=1),
                                                 tf.split(outputs.nodes, self.batch, axis=1))]
        return loss_ops


    def training(self):
        next_batch = load_tfrecord()
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), module=[self.generator, self.discriminator])
        manager = tf.train.CheckpointManager(checkpoint, self.model_path, max_to_keep=3)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        train_dataset = next_batch.getDataset(self.train_data, self.batch, True)
        # space_height, space_width, filename, inputs, outputs = next(iter(train_dataset))

        self.step = 0
        for epoch in range(self.epochs):
            print("\nStart of epoch %d" % epoch)
            for step, (space_height, space_width, filename, inputs, outputs) in enumerate(train_dataset):
                step_pre_batch = len(filename)
                ############# training step ##############
                checkpoint.step.assign_add(1)
                input_tuple = self.graphTuple(inputs, 'inputs')
                target_tuple = self.graphTuple(outputs, 'target')
                gen_loss, disc_loss = self.train_step(input_tuple, target_tuple)
                if step % 200 == 0:
                    print("Training loss (for %d batch) at step %d: %.8f, %.8f"
                          % (int(step_pre_batch), step, float(gen_loss), float(disc_loss)),
                          "samples: {}".format(filename[-1]),
                          "lr_rate: {:0.6f}".format(self.generator_opt._decayed_lr(tf.float32).numpy()))
                    self.step += 200

                # if int(step) % 1001 == 0:
                #     ############# save validatoin plot image #############
                #     space_height, space_width, filename, inputs, outputs = next(iter(train_dataset))
                #     input_tuple = self.graphTuple(inputs, 'inputs')
                #     generated_output = self.graphTuple(outputs, 'target')
                #     self.plotStep(
                #         input_tuple,
                #         generated_output,
                #         np.array(space_height, dtype=np.int16)[0],
                #         np.array(space_width, dtype=np.int16)[0],
                #         str(self.step) + '_' + np.array(filename)[0].decode())

                    ############# save checkpoint ##############
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

                if int(step) % 10000 == 0:
                    ############# save model ##############
                    to_save = snt.Module()
                    to_save.inference = self.predict_step
                    to_save.all_variables = list(self.generator.variables)
                    tf.saved_model.save(to_save, self.model_path)
                    print("Saved module for step {}".format(int(step)))

    def validation(self):
        next_batch = load_tfrecord()
        test_dataset = next_batch.getDataset(self.train_data, self.batch, True)

        for epoch in range(1):
            print("\nStart of epoch %d" % epoch)
            for step, (space_height, space_width, filename, inputs, outputs) in enumerate(test_dataset):
                input_tuple = self.graphTuple(inputs, 'inputs')
                generated_output = self.graphTuple(outputs, 'target')
                self.plotTarget(
                    input_tuple,
                    generated_output,
                    np.array(space_height, dtype=np.int16)[0],
                    np.array(space_width, dtype=np.int16)[0],
                    'target_' + np.array(filename)[0].decode())


    def test(self, csv_file):
        next_batch = load_tfrecord()
        test_dataset = next_batch.customDataset(csv_file)
        iteration = input("출력할 횟수: ")

        for epoch in range(int(iteration)):
            print("\nStart of epoch %d" % epoch)
            for step, (space_height, space_width, filename, inputs) in enumerate(test_dataset):
                input_tuple = self.graphTuple([inputs], 'inputs')
                generated_output = self.predict_step(input_tuple)
                self.plotStep(
                    input_tuple,
                    generated_output,
                    np.array(space_height, dtype=np.int16),
                    np.array(space_width, dtype=np.int16),
                    'generate_' + filename)



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

    # def get_predicted(
    #         self,
    #         position_sequence,
    #         n_particles_per_example,
    #         particle_types=None
    # ):
    #
    #     input_graphs_tuple = self._encoder_preprocessor(
    #         position_sequence,
    #         n_particles_per_example,
    #         particle_types
    #     )
    #
    #     predicted_velocity = self.graph_networks(input_graphs_tuple, self._num_processing_steps)
    #     next_position = _decoder_postprocessor(normalized_velocity[-1].nodes, position_sequence)
    #     return next_position

    def _encoder_preprocessor(
            self,
            position_sequence,
            n_node,
            particle_types
    ):
        # boundary_sequence, n_bound = self._boundary_sequence(position_sequence[0, -1])
        # position_boundary = tf.concat([position_sequence, boundary_sequence], axis=0)
        # n_nodes = n_node + n_bound

        (senders, receivers, n_edge) = connectivity_utils.compute_connectivity_for_batch_pyfunc(
            position_sequence, n_node, self._connectivity_radius
        )

        node_features = []
        flat_position_sequence = snt.reshape(position_sequence, output_shape=(-1,))
        node_features.append(flat_position_sequence)

        edge_features = []

        normalized_relative_displacements = (tf.gather(position_sequence, senders) -
                                             tf.gather(position_sequence, receivers)) / self._connectivity_radius
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

    def get_predicted_velocity(
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

        return self.graph_networks(input_graphs_tuple, self._num_processing_steps)

    def _inverse_decoder_postprocessor(
            self,
            next_position,
            position_sequence):
        previous_position = position_sequence[:, -1]
        velocity = next_position - previous_position
        return velocity

    def boundary_sequence(self, times):
        const = tf.constant(self._boundaries, dtype=tf.float32)
        times = tf.linspace(times, times, self.num_boundaries)[:, tf.newaxis]
        x_min_min = tf.linspace(const[0, 0], const[0, 0], self.num_boundaries)[:, tf.newaxis]
        x_max_max = tf.linspace(const[0, 1], const[0, 1], self.num_boundaries)[:, tf.newaxis]
        x_min_max = tf.linspace(const[0, 0], const[0, 1], self.num_boundaries)[:, tf.newaxis]
        y_min_min = tf.linspace(const[1, 0], const[1, 0], self.num_boundaries)[:, tf.newaxis]
        y_max_max = tf.linspace(const[1, 1], const[1, 1], self.num_boundaries)[:, tf.newaxis]
        y_min_max = tf.linspace(const[1, 0], const[1, 1], self.num_boundaries)[:, tf.newaxis]
        boundaries = tf.concat([
            tf.concat([x_min_max, y_min_min, times], axis=1),
            tf.concat([x_min_max, y_max_max, times], axis=1),
            tf.concat([x_min_min, y_min_max, times], axis=1),
            tf.concat([x_max_max, y_min_max, times], axis=1)
        ], axis=0)

        n_boundary = boundaries.get_shape().as_list()[0]
        return boundaries, n_boundary

    def boundary_velocity(self):
        zeros = tf.zeros([self.num_boundaries * 4, 2])
        return zeros
