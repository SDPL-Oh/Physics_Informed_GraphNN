import graph_nets as gn
import sonnet as snt


def build_mlp(
        hidden_size: int, num_hidden_layers: int, output_size: int) -> snt.Module:
    return snt.nets.MLP(
        output_sizes=[hidden_size] * num_hidden_layers + [output_size])


def build_mlp_with_layer_norm():
    mlp = build_mlp(
        hidden_size=128,
        num_hidden_layers=2,
        output_size=128)
    return snt.Sequential([mlp, snt.LayerNorm(axis=-1, create_offset=True, create_scale=True)])


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
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm,
            global_model_fn=build_mlp_with_layer_norm
        )

    def __call__(self, inputs):
        return self._network(inputs)


class GraphDecoder(snt.Module):
    def __init__(self, name="GraphDecoder"):
        super(GraphDecoder, self).__init__(name=name)
        self._network = gn.modules.GraphIndependent(
            edge_model_fn=build_mlp_with_layer_norm,
            node_model_fn=build_mlp_with_layer_norm,
            global_model_fn=build_mlp_with_layer_norm)

    def __call__(self, inputs):
        return self._network(inputs)


class GraphCore(snt.Module):
    def __init__(self, name="GraphCore"):
        super(GraphCore, self).__init__(name=name)
        self._network = gn.modules.GraphNetwork(
            build_mlp_with_layer_norm,
            build_mlp_with_layer_norm,
            build_mlp_with_layer_norm
        )

    def __call__(self, inputs):
        return self._network(inputs)
