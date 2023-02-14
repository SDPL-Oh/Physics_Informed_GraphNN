import tensorflow as tf
import graph_nets as gn
import sonnet as snt


def make_mlp_model():
    return snt.Sequential([
        snt.Linear(128),
        tf.keras.layers.LeakyReLU(),
        snt.Linear(128),
        tf.keras.layers.LeakyReLU(),
        snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)
    ])


class GraphEPD(snt.Module):
    def __init__(self,
                 edge_output_size=None,
                 node_output_size=None,
                 global_output_size=None,
                 name="GraphEPD"):
        super(GraphEPD, self).__init__(name=name)
        self._encoder = GraphEncoder()
        self._core = GraphCore()
        self._second_core = GraphCore()
        self._decoder = GraphDecoder()
        if edge_output_size is None:
            edge_fn = None
        else:
            edge_fn = lambda: snt.Linear(edge_output_size, name="edge_output")
        if node_output_size is None:
            node_fn = None
        else:
            node_fn = lambda: snt.Linear(node_output_size, name="node_output")
        if global_output_size is None:
            global_fn = None
        else:
            global_fn = lambda: snt.Linear(global_output_size, name="global_output")
        self._output_transform = gn.modules.GraphIndependent(
            edge_fn, node_fn, global_fn)

    def __call__(self, input_op, num_processing_steps):
        latent = self._encoder(input_op)
        latent0 = latent
        output_ops = []
        for _ in range(num_processing_steps):
            core_input = gn.utils_tf.concat([latent0, latent], axis=1)
            latent = self._core(core_input)
            decoded_op = self._decoder(latent)
            output_ops.append(self._output_transform(decoded_op))
        return output_ops


class GraphEncoder(snt.Module):
    def __init__(self, name="GraphEncoder"):
        super(GraphEncoder, self).__init__(name=name)
        self._network = gn.modules.GraphIndependent(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model
        )

    def __call__(self, inputs):
        return self._network(inputs)


class GraphDecoder(snt.Module):
    def __init__(self, name="GraphDecoder"):
        super(GraphDecoder, self).__init__(name=name)
        self._network = gn.modules.GraphIndependent(
            edge_model_fn=make_mlp_model,
            node_model_fn=make_mlp_model,
            global_model_fn=make_mlp_model)

    def __call__(self, inputs):
        return self._network(inputs)


class GraphCore(snt.Module):
    def __init__(self, name="GraphCore"):
        super(GraphCore, self).__init__(name=name)
        self._network = gn.modules.GraphNetwork(
            make_mlp_model,
            make_mlp_model,
            make_mlp_model
        )

    def __call__(self, inputs):
        return self._network(inputs)
