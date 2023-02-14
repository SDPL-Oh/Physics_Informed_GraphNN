import collections
import numpy as np
import tensorflow as tf

import noise_utils
import learned_simulator

"""
Particle type: 다양한 재료의 특성을 반영할 수 있도록 class 로 설정함.
Number of Particle types: 현재 적용 가능한 재료 특성 종류의 수. 데이터 라벨링으로 지정되어 있음
                            Water, Sand, Goop(젤리 같은 재질), rigid, boundary
"""

INPUT_SEQUENCE_LENGTH = 6
SEQUENCE_LENGTH = INPUT_SEQUENCE_LENGTH + 1  # add one target position.
NUM_DIMENSIONS = 3
NUM_PARTICLE_TYPES = 6
BATCH_SIZE = 5
GLOBAL_CONTEXT_SIZE = 6
NUM_PROCESSING_STEPS = 5

Stats = collections.namedtuple("Stats", ["mean", "std"])

DUMMY_STATS = Stats(
    mean=np.zeros([NUM_DIMENSIONS], dtype=np.float32),
    std=np.ones([NUM_DIMENSIONS], dtype=np.float32))
DUMMY_CONTEXT_STATS = Stats(
    mean=np.zeros([GLOBAL_CONTEXT_SIZE], dtype=np.float32),
    std=np.ones([GLOBAL_CONTEXT_SIZE], dtype=np.float32))
DUMMY_BOUNDARIES = [(-1., 1.)] * NUM_DIMENSIONS  # [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]


def sample_random_position_sequence():
    """
    무작위로 particle 을 생성하여 sequence 를 구성함: 입력 노드 수에 관계없이 그래프를 일반화하여 학습하는 방법
    Returns mock data mimicking the input features collected by the encoder.
    """
    num_particles = tf.random.uniform(shape=(), minval=50, maxval=1000, dtype=tf.int32)
    position_sequence = tf.random.normal(shape=[num_particles, SEQUENCE_LENGTH, NUM_DIMENSIONS])
    return position_sequence


def main():
    # Build the model.
    learnable_model = learned_simulator.LearnedSimulator(
        num_dimensions=NUM_DIMENSIONS,
        connectivity_radius=0.05,
        boundaries=DUMMY_BOUNDARIES,
        normalization_stats={"acceleration": DUMMY_STATS, "velocity": DUMMY_STATS, "context": DUMMY_CONTEXT_STATS, },
        num_particle_types=NUM_PARTICLE_TYPES,
        num_processing_steps=NUM_PROCESSING_STEPS,
        particle_type_embedding_size=16,
    )

    # ex) [(974, 7, 3), (84, 7, 3), (846, 7, 3), (348, 7, 3), (474, 7, 3)]
    sampled_position_sequences = [sample_random_position_sequence() for _ in range(BATCH_SIZE)]
    # ex) (2726, 7, 3)
    position_sequence_batch = tf.concat(sampled_position_sequences, axis=0)
    # ex) [974, 84, 846, 348, 474]
    n_particles_per_example = tf.stack([tf.shape(seq)[0] for seq in sampled_position_sequences], axis=0)
    # ex) (2726, ) 여기서는 무작위로 각 노드마다 particle type 을 지정함.
    particle_types = tf.random.uniform([tf.shape(position_sequence_batch)[0]], 0, NUM_PARTICLE_TYPES, dtype=tf.int32)
    # ex) (5, 6) 여기서는 무작위로 각 batch 마다 global context 를 지정함.
    global_context = tf.random.uniform([BATCH_SIZE, GLOBAL_CONTEXT_SIZE], -1., 1., dtype=tf.float32)
    # ex) (2726, 6, 3)
    input_position_sequence = position_sequence_batch[:, :-1]
    # ex) (2726, 3)
    target_next_position = position_sequence_batch[:, -1]
    # ex) (2726, 3)
    predicted_next_position = learnable_model.build(
        input_position_sequence, n_particles_per_example, global_context, particle_types
    )
    print(f"Per-particle output tensor: {predicted_next_position.shape}")
    # (2726, 6, 3)
    position_sequence_noise = (
        noise_utils.get_random_walk_noise_for_position_sequence(input_position_sequence, noise_std_last_step=6.7e-4))

    # ex) (2726, 3), (2726, 3)
    predicted_normalized_acceleration, target_normalized_acceleration = (
        learnable_model.get_predicted_and_target_normalized_accelerations(
            next_position=target_next_position,
            position_sequence=input_position_sequence,
            position_sequence_noise=position_sequence_noise,
            n_particles_per_example=n_particles_per_example,
            particle_types=particle_types,
            global_context=global_context,

        )
    )

    print(f"Predicted norm. acceleration: {predicted_normalized_acceleration[-1].nodes.shape}")
    print(f"Target norm. acceleration: {target_normalized_acceleration.shape}")


# import os
# import json
# import functools
# import tree
# from graph_pinn.utils import reading_utils
#
#
# def _read_metadata(data_path):
#     with open(os.path.join(data_path, 'metadata.json'), 'rt') as fp:
#         return json.loads(fp.read())
#
#
# def prepare_inputs(tensor_dict):
#     """Prepares a single stack of inputs by calculating inputs and targets.
#
#   Computes n_particles_per_example, which is a tensor that contains information
#   about how to partition the axis - i.e. which nodes belong to which graph.
#
#   Adds a batch axis to `n_particles_per_example` and `step_context` so they can
#   later be batched using `batch_concat`. This batch will be the same as if the
#   elements had been batched via stacking.
#
#   Note that all other tensors have a variable size particle axis,
#   and in this case they will simply be concatenated along that
#   axis.
#
#
#
#   Args:
#     tensor_dict: A dict of tensors containing positions, and step context (
#     if available).
#
#   Returns:
#     A tuple of input features and target positions.
#
#   """
#     # Position is encoded as [sequence_length, num_particles, dim] but the model
#     # expects [num_particles, sequence_length, dim].
#     pos = tensor_dict['position']
#     pos = tf.transpose(pos, perm=[1, 0, 2])
#
#     # The target position is the final step of the stack of positions.
#     target_position = pos[:, -1]
#
#     # Remove the target from the input.
#     tensor_dict['position'] = pos[:, :-1]
#
#     # Compute the number of particles per example.
#     num_particles = tf.shape(pos)[0]
#     # Add an extra dimension for stacking via concat.
#     tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]
#
#     if 'step_context' in tensor_dict:
#         # Take the input global context. We have a stack of global contexts,
#         # and we take the penultimate since the final is the target.
#         tensor_dict['step_context'] = tensor_dict['step_context'][-2]
#         # Add an extra dimension for stacking via concat.
#         tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
#     return tensor_dict, target_position
#
#
# def batch_concat(dataset, batch_size):
#     """We implement batching as concatenating on the leading axis."""
#
#     # We create a dataset of datasets of length batch_size.
#     windowed_ds = dataset.window(batch_size)
#
#     # The plan is then to reduce every nested dataset by concatenating. We can
#     # do this using tf.data.Dataset.reduce. This requires an initial state, and
#     # then incrementally reduces by running through the dataset
#
#     # Get initial state. In this case this will be empty tensors of the
#     # correct shape.
#     initial_state = tree.map_structure(
#         lambda spec: tf.zeros(  # pylint: disable=g-long-lambda
#             shape=[0] + spec.shape.as_list()[1:], dtype=spec.dtype),
#         dataset.element_spec)
#
#     def reduce_window(initial_state, ds):
#         return ds.reduce(initial_state, lambda x, y: tf.concat([x, y], axis=0))
#
#     return windowed_ds.map(
#         lambda *x: tree.map_structure(reduce_window, initial_state, x))
#
#
# def prepare_rollout_inputs(context, features):
#     """Prepares an inputs trajectory for rollout."""
#     out_dict = {**context}
#     pos = tf.transpose(features['position'], [1, 0, 2])
#     target_position = pos[:, -1]
#     # Remove the target from the input.
#     out_dict['position'] = pos[:, :-1]
#     # Compute the number of nodes
#     out_dict['n_particles_per_example'] = [tf.shape(pos)[0]]
#     if 'step_context' in features:
#         out_dict['step_context'] = features['step_context']
#     out_dict['is_trajectory'] = tf.constant([True], tf.bool)
#     return out_dict, target_position
#
#
# def input_fn(self, data_path, batch_size, mode, split):
#     metadata = _read_metadata(self.data_path)
#     ds = tf.data.TFRecordDataset([os.path.join(self.data_path, f'{split}.tfrecord')])
#     ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
#     if mode.startswith('one_step'):
#         split_with_window = functools.partial(
#             reading_utils.split_trajectory, window_length=INPUT_SEQUENCE_LENGTH + 1
#         )
#         ds = ds.flat_map(split_with_window)
#         ds = ds.map(prepare_inputs)
#         if mode == 'one_step_train':
#             ds = ds.repeat()
#             ds = ds.shuffle(512)
#         batch_concat(ds, self.batch_size)
#
#     elif mode == 'rollout':
#         assert self.batch_size == 1
#         ds = ds.map(prepare_rollout_inputs)
#     else:
#         raise ValueError(f'mode: {mode} not recognized')
#     return ds


# a = GetInput("F:/PhysicalProject/graph_pinn/WaterDropSample", 1)
# datat = a.input_fn("rollout", "valid")
# print(next(iter(datat)))

if __name__ == "__main__":
    main()
