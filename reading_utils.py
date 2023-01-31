import functools
import numpy as np
import tensorflow as tf

_FEATURE_DESCRIPTION = {'position': tf.io.VarLenFeature(tf.string),}
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT = _FEATURE_DESCRIPTION.copy()
_FEATURE_DESCRIPTION_WITH_GLOBAL_CONTEXT['step_context'] = tf.io.VarLenFeature(tf.string)

_FEATURE_DTYPES = {
    'position': {
        'in': np.float32,
        'out': tf.float32
    },
    'step_context': {
        'in': np.float32,
        'out': tf.float32
    }
}

_CONTEXT_FEATURES = {
    # 'key': tf.io.FixedLenFeature([], tf.int64, default_value=0),
    'particle_type': tf.io.VarLenFeature(tf.string)
}


def convert_to_tensor(x, encoded_dtype):
    if len(x) == 1:
        out = np.frombuffer(x[0].numpy(), dtype=encoded_dtype)
    else:
        out = []
    for el in x:
        out = np.append(out, np.frombuffer(el.numpy(), dtype=encoded_dtype))
    out = tf.cast(np.array(out), dtype=encoded_dtype)
    return out


def parse_serialized_simulation_example(example_proto, metadata):
    feature_description = _FEATURE_DESCRIPTION
    context, parsed_features = tf.io.parse_single_sequence_example(
        example_proto,
        context_features=_CONTEXT_FEATURES,
        sequence_features=feature_description
    )
    for feature_key, item in parsed_features.items():
        convert_fn = functools.partial(
            convert_to_tensor, encoded_dtype=_FEATURE_DTYPES[feature_key]['in']
        )
        parsed_features[feature_key] = tf.py_function(
            convert_fn, inp=[item.values], Tout=_FEATURE_DTYPES[feature_key]['out']
        )
    position_shape = [metadata['sequence_length'] + 1, -1, metadata['dim']]
    parsed_features['position'] = tf.reshape(parsed_features['position'], position_shape)

    # time_sequence = tf.range(metadata['sequence_length'] + 1, dtype=tf.float32)
    # parsed_features['time'] = tf.reshape(time_sequence, [metadata['sequence_length'] + 1, -1, metadata['dim']])
    # print(parsed_features['time'])

    context['particle_type'] = tf.py_function(
        functools.partial(convert_to_tensor, encoded_dtype=np.int64),
        inp=[context['particle_type'].values],
        Tout=[tf.int64])
    context['particle_type'] = tf.reshape(context['particle_type'], [-1])

    return context, parsed_features


def split_trajectory(context, features, window_length=2):
    trajectory_length = features['position'].get_shape().as_list()[0]
    features['time'] = tf.linspace(0, 1, trajectory_length)[:, tf.newaxis]
    input_trajectory_length = trajectory_length - window_length + 1
    model_input_features = {'particle_type': tf.tile(
        tf.expand_dims(context['particle_type'], axis=0), [input_trajectory_length, 1])}

    pos_stack, time_stack = [], []
    for idx in range(input_trajectory_length):
        pos_stack.append(features['position'][idx:idx + window_length])
        time_stack.append(features['time'][idx:idx + window_length])
    model_input_features['position'] = tf.stack(pos_stack)
    model_input_features['time'] = tf.stack(time_stack)

    return tf.data.Dataset.from_tensor_slices(model_input_features)
