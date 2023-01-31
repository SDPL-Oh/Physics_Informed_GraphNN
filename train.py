import collections
import functools
import json
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pickle

from absl import app
from absl import flags
from absl import logging
import numpy as np
import tensorflow as tf
import sonnet as snt
import tree

import learned_simulator
import noise_utils
import reading_utils

flags.DEFINE_enum(
    'mode',
    'eval_rollout',
    ['train', 'eval', 'eval_rollout'],
    help='Train model, one step evaluation or rollout evaluation.'
)
flags.DEFINE_enum(
    'eval_split',
    'test',
    ['train', 'valid', 'test'],
    help='Split to use when running evaluation.'
)
flags.DEFINE_string(
    'data_path',
    'WaterDropSample',
    help='The dataset directory.'
)
flags.DEFINE_integer(
    'batch_size',
    1,
    help='The batch size.'
)
flags.DEFINE_integer(
    'num_steps',
    int(2e7),
    help='Number of steps of training.'
)
flags.DEFINE_float(
    'noise_std',
    6.7e-4,
    help='The std deviation of the noise.'
)
flags.DEFINE_string(
    'model_path',
    'models',
    help='The path for saving checkpoints of the model. Defaults to a temporary directory.'
)
flags.DEFINE_string(
    'output_path',
    'rollout/WaterDropSample/',
    help='The path for saving outputs (e.g. rollouts).'
)

FLAGS = flags.FLAGS

Stats = collections.namedtuple('Stats', ['mean', 'std'])

INPUT_SEQUENCE_LENGTH = 2  # So we can calculate the last 5 velocities.
NUM_PARTICLE_TYPES = 9
KINEMATIC_PARTICLE_ID = 3
NUM_PROCESSING_STEPS = 2
Re = 100


def get_kinematic_mask(particle_types):
    return tf.equal(particle_types, KINEMATIC_PARTICLE_ID)


def prepare_inputs(tensor_dict):
    pos = tensor_dict['position']
    pos = tf.transpose(pos, perm=[1, 0, 2])

    tensor_dict['n_particles_per_example'] = tf.shape(pos)[0][tf.newaxis]
    seq_time = tf.tile(tf.expand_dims(tensor_dict['time'], axis=0), [tf.shape(pos)[0], 1, 1])
    tensor_dict['position'] = tf.concat([pos, tf.cast(seq_time, tf.float32)], axis=-1)

    target_position = tensor_dict['position'][:, -1]

    return tensor_dict, target_position


def prepare_rollout_inputs(features):
    out_dict = {}
    pos = tf.transpose(features['position'], [1, 0, 2])
    target_position = pos[:, -1]
    out_dict['position'] = pos[:, :-1]
    out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]

    out_dict['is_trajectory'] = tf.constant([True], tf.bool)
    return out_dict, target_position


def batch_concat(dataset, batch_size):
    windowed_ds = dataset.window(batch_size)
    initial_state = tree.map_structure(
        lambda spec: tf.zeros(shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
        dataset.element_spec
    )

    def reduce_window(initial_state, ds):
        return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))

    return windowed_ds.map(
        lambda *x: tree.map_structure(reduce_window, initial_state, x))


def input_fn(data_path, batch_size, mode, split):
    metadata = _read_metadata(data_path)
    ds = tf.data.TFRecordDataset([os.path.join(data_path, f'{split}.tfrecord')])
    ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
    if mode.startswith('one_step'):
        split_with_window = functools.partial(
            reading_utils.split_trajectory, window_length=INPUT_SEQUENCE_LENGTH)
        ds = ds.flat_map(split_with_window)
        ds = ds.map(prepare_inputs)
        if mode == 'one_step_train':
            ds = ds.repeat()
            ds = ds.shuffle(512)
        ds = batch_concat(ds, batch_size)
    elif mode == 'rollout':
        ds = ds.map(prepare_rollout_inputs)
    else:
        raise ValueError(f'mode: {mode} not recognized')
    return ds


def _read_metadata(data_path):
    with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
        return json.loads(fp.read())


def _combine_std(std_x, std_y):
    return np.sqrt(std_x ** 2 + std_y ** 2)


class LearningSimulator:
    def __init__(self, data_path, acc_noise_std, vel_noise_std):
        self.metadata = _read_metadata(data_path)
        self.acc_noise_std = acc_noise_std
        self.vel_noise_std = vel_noise_std
        self.simulator = self.get_simulator()

        self.lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            1e-6,
            decay_steps=50000,
            decay_rate=0.1,
            staircase=True
        )

        self.opt = tf.keras.optimizers.Adam(learning_rate=self.lr_schedule)

    def get_simulator(self):
        simulator = learned_simulator.LearnedSimulator(
            num_dimensions=self.metadata['dim'],
            connectivity_radius=self.metadata['default_connectivity_radius'],
            boundaries=self.metadata['bounds'],
            num_boundaries=100,
            num_particle_types=NUM_PARTICLE_TYPES,
            num_processing_steps=NUM_PROCESSING_STEPS,
            particle_type_embedding_size=16)
        return simulator

    def rollout(self, features, num_steps):
        initial_positions = features['position'][:, 0:INPUT_SEQUENCE_LENGTH]
        ground_truth_positions = features['position'][:, INPUT_SEQUENCE_LENGTH:]
        global_context = features.get('step_context')

        def step_fn(step, current_positions, predictions):
            if global_context is None:
                global_context_step = None
            else:
                global_context_step = global_context[
                    step + INPUT_SEQUENCE_LENGTH - 1][tf.newaxis]

            next_position = self.simulator.build(
                current_positions,
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_type'][:tf.shape(initial_positions)[0]],
                global_context=global_context_step
            )

            kinematic_mask = get_kinematic_mask(
                features['particle_type'][:tf.shape(initial_positions)[0]][:, tf.newaxis]
            )
            next_position_ground_truth = ground_truth_positions[:, step]

            next_position = tf.where(kinematic_mask, next_position_ground_truth, next_position)
            updated_predictions = predictions.write(step, next_position)

            next_positions = tf.concat([current_positions[:, 1:], next_position[:, tf.newaxis]], axis=1)

            return step + 1, next_positions, updated_predictions

        predictions = tf.TensorArray(size=num_steps, dtype=tf.float32)
        _, _, predictions = tf.while_loop(
            cond=lambda step, state, prediction: tf.less(step, num_steps),
            body=step_fn,
            loop_vars=(0, initial_positions, predictions),
            back_prop=False,
            parallel_iterations=1)

        output_dict = {
            'initial_positions': tf.transpose(initial_positions, [1, 0, 2]),
            'predicted_rollout': predictions.stack(),
            'ground_truth_rollout': tf.transpose(ground_truth_positions, [1, 0, 2]),
            'particle_types': features['particle_type'],
        }

        if global_context is not None:
            output_dict['global_context'] = global_context

        return output_dict

    def second_ns_net(self, features, labels, t):
        target_next_position = labels

        sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
            features['position'], noise_std_last_step=self.acc_noise_std)

        non_kinematic_mask = tf.logical_not(
            get_kinematic_mask(features['particle_type'][:tf.shape(sampled_noise)[0]]))
        noise_mask = tf.cast(
            non_kinematic_mask, sampled_noise.dtype)[:, tf.newaxis, tf.newaxis]
        sampled_noise *= noise_mask

        pred_target = self.simulator.get_predicted_and_target_velocity(
            next_position=target_next_position,
            position_sequence=features['position'],
            position_sequence_noise=sampled_noise,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_type'][:tf.shape(sampled_noise)[0]],
            time_sequence=features['time']
        )

        pred_velocity, target_velocity = pred_target

        out = self.model(inputs)
        phi = out[:, 0]
        p = out[:, 1]
        g = tf.Graph()
        with g.as_default():
            u = tf.gradients(phi, y)[0]
            v = - tf.gradients(phi, x)[0]

            u_t = tf.gradients(u, t)[0]
            u_x = tf.gradients(u, x)[0]
            u_y = tf.gradients(u, y)[0]
            u_xx = tf.gradients(u_x, x)[0]
            u_yy = tf.gradients(u_y, y)[0]

            v_t = tf.gradients(v, t)[0]
            v_x = tf.gradients(v, x)[0]
            v_y = tf.gradients(v, y)[0]
            v_xx = tf.gradients(v_x, x)[0]
            v_yy = tf.gradients(v_y, y)[0]

            p_x = tf.gradients(p, x)[0]
            p_y = tf.gradients(p, y)[0]

        f = u_t + u * u_x + v * u_y + p_x - 1 / Re * (u_xx + u_yy)
        g = v_t + u * v_x + v * v_y + p_y - 1 / Re * (v_xx + v_yy)

        return u, v, p, f, g

    @tf.function
    def train_step(self, features, labels):
        with tf.GradientTape() as tape:
            target_next_position = labels

            sampled_noise = noise_utils.get_random_walk_noise_for_position_sequence(
                features['position'], noise_std_last_step=self.acc_noise_std)

            non_kinematic_mask = tf.logical_not(
                get_kinematic_mask(features['particle_type'][:tf.shape(sampled_noise)[0]]))
            noise_mask = tf.cast(
                non_kinematic_mask, sampled_noise.dtype)[:, tf.newaxis, tf.newaxis]
            sampled_noise *= noise_mask

            pred_target = self.simulator.get_predicted_and_target_velocity(
                next_position=target_next_position,
                position_sequence=features['position'],
                position_sequence_noise=sampled_noise,
                n_particles_per_example=features['n_particles_per_example'],
                particle_types=features['particle_type'][:tf.shape(sampled_noise)[0]],
                global_context=features.get('step_context')
            )

            pred_velocity, target_velocity = pred_target

            loss = (pred_velocity[-1].nodes - target_velocity) ** 2

            non_kinematic_mask_dim = non_kinematic_mask[:, tf.newaxis]
            num_non_kinematic = tf.reduce_sum(tf.cast(non_kinematic_mask, tf.float32))
            loss = tf.where(non_kinematic_mask_dim, loss, tf.zeros_like(loss))
            loss = tf.reduce_sum(loss) / tf.reduce_sum(num_non_kinematic)

        gradients = tape.gradient(loss, self.simulator.graph_networks.trainable_variables)
        self.opt.apply_gradients(zip(gradients, self.simulator.graph_networks.trainable_variables))

        return loss

    def training(self):
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), module=[self.simulator])
        manager = tf.train.CheckpointManager(checkpoint, FLAGS.model_path, max_to_keep=3)
        train_summary_writer = tf.summary.create_file_writer(FLAGS.model_path)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        ds = input_fn(FLAGS.data_path, FLAGS.batch_size, mode='one_step_train', split='train')
        for epoch in range(FLAGS.num_steps):
            print("\nStart of epoch %d" % epoch)
            for step, (features, labels) in enumerate(ds):
                checkpoint.step.assign_add(1)
                loss = self.train_step(features, labels)

                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss, step=step)

                if step % 200 == 0:
                    print("[Training loss] step %d: %.8f"
                          % (step, float(loss)),
                          "lr_rate: {:0.6f}".format(self.opt._decayed_lr(tf.float32).numpy()))

                    ############# save checkpoint ##############
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(checkpoint.step), save_path))

                if int(step) % 10000 == 0:
                    ############# save model ##############
                    to_save = snt.Module()
                    to_save.inference = self.predict_step
                    to_save.all_variables = list(self.simulator.graph_networks.variables)
                    tf.saved_model.save(to_save, FLAGS.model_path)
                    print("Saved module for step {}".format(int(step)))

    @tf.function
    def predict_step(self, features):
        num_steps = self.metadata['sequence_length'] - INPUT_SEQUENCE_LENGTH
        rollout_op = self.rollout(features, num_steps=num_steps)
        return rollout_op

    @tf.function
    def test_step(self, features, labels):
        sampled_noise = features['position']

        non_kinematic_mask = tf.logical_not(
            get_kinematic_mask(features['particle_type'][:tf.shape(sampled_noise)[0]]))
        noise_mask = tf.cast(
            non_kinematic_mask, sampled_noise.dtype)[:, tf.newaxis, tf.newaxis]
        sampled_noise *= noise_mask

        pred_target = self.simulator.get_predicted_and_target_velocity(
            position_sequence=features['position'],
            position_sequence_noise=sampled_noise,
            n_particles_per_example=features['n_particles_per_example'],
            particle_types=features['particle_type'][:tf.shape(sampled_noise)[0]]
        )

        print(pred_target)

    def validation(self):
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), module=[self.simulator])
        manager = tf.train.CheckpointManager(checkpoint, FLAGS.model_path, max_to_keep=3)
        if manager.latest_checkpoint:
            checkpoint.restore(manager.latest_checkpoint)
            print("Restored from {}".format(manager.latest_checkpoint))
        else:
            print("Initializing from scratch.")

        features = input_fn(FLAGS.data_path, FLAGS.batch_size, mode='rollout', split=FLAGS.eval_split)

        for epoch in range(1):
            for step, (feature, _) in enumerate(features):
                predicted_next_position = self.predict_step(feature)
                predicted_next_position['metadata'] = self.metadata
                filename = f'rollout_{FLAGS.eval_split}_{step}.pkl'
                filename = os.path.join(FLAGS.output_path, filename)
                logging.info('Saving: %s.', filename)
                if not os.path.exists(FLAGS.output_path):
                    os.mkdir(FLAGS.output_path)
                with open(filename, 'wb') as file:
                    pickle.dump(predicted_next_position, file)
                print(step)

    def test_system(self):
        ds = input_fn(FLAGS.data_path, FLAGS.batch_size, mode='one_step_train', split='train')
        for epoch in range(FLAGS.num_steps):
            print("\nStart of epoch %d" % epoch)
            for step, (features, labels) in enumerate(ds):
                self.test_step(features, labels)


def main(_):
    if FLAGS.mode in ['train', 'eval_rollout']:
        learning = LearningSimulator(FLAGS.data_path, FLAGS.noise_std, FLAGS.noise_std)
        # if FLAGS.mode == 'train':
        #     learning.training()
        # else:
        #     learning.validation()

        learning.test_system()


if __name__ == '__main__':
    app.run(main)
