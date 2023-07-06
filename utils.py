#!/usr/bin/env python3

import os
import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.modules import LSTMModule, MLP
from vocabulary import DeNovoVocabulary


class Embed(torch.nn.Module):
    """Implements a simple embedding layer."""

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self._embedding = torch.nn.Embedding(input_size, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, L = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


def create_model(vocabulary, output_size, out_key="logits"):

    embedding_module = TensorDictModule(
        Embed(len(vocabulary), 256),
        in_keys=["obs"],
        out_keys=["embed"],
    )
    lstm_module = LSTMModule(
        input_size=256,
        hidden_size=512,
        num_layers=3,
        in_key="embed",
        out_key="features",
    )
    mlp = TensorDictModule(
        MLP(
            in_features=512,
            out_features=output_size,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    return TensorDictSequential(embedding_module, lstm_module.set_recurrent_mode(True), mlp)


def create_rhs_transform():
    lstm_module = LSTMModule(
        input_size=256,
        hidden_size=512,
        num_layers=3,
        in_key="embed",
        out_key="features",
    )
    return lstm_module.make_tensordict_primer()


reinvent_weights_policy_mapping = {
    "_embedding.weight": "module.0.module._embedding.weight",
    "_rnn.weight_ih_l0": "module.1.lstm.weight_ih_l0",
    "_rnn.weight_hh_l0": "module.1.lstm.weight_hh_l0",
    "_rnn.bias_ih_l0": "module.1.lstm.bias_ih_l0",
    "_rnn.bias_hh_l0": "module.1.lstm.bias_hh_l0",
    "_rnn.weight_ih_l1": "module.1.lstm.weight_ih_l1",
    "_rnn.weight_hh_l1": "module.1.lstm.weight_hh_l1",
    "_rnn.bias_ih_l1": "module.1.lstm.bias_ih_l1",
    "_rnn.bias_hh_l1": "module.1.lstm.bias_hh_l1",
    "_rnn.weight_ih_l2": "module.1.lstm.weight_ih_l2",
    "_rnn.weight_hh_l2": "module.1.lstm.weight_hh_l2",
    "_rnn.bias_ih_l2": "module.1.lstm.bias_ih_l2",
    "_rnn.bias_hh_l2": "module.1.lstm.bias_hh_l2",
    "_linear.weight": "module.2.module.0.weight",
    "_linear.bias": "module.2.module.0.bias",
}

reinvent_weights_value_mapping = {
    "_embedding.weight": "module.0.module._embedding.weight",
    "_rnn.weight_ih_l0": "module.1.lstm.weight_ih_l0",
    "_rnn.weight_hh_l0": "module.1.lstm.weight_hh_l0",
    "_rnn.bias_ih_l0": "module.1.lstm.bias_ih_l0",
    "_rnn.bias_hh_l0": "module.1.lstm.bias_hh_l0",
    "_rnn.weight_ih_l1": "module.1.lstm.weight_ih_l1",
    "_rnn.weight_hh_l1": "module.1.lstm.weight_hh_l1",
    "_rnn.bias_ih_l1": "module.1.lstm.bias_ih_l1",
    "_rnn.bias_hh_l1": "module.1.lstm.bias_hh_l1",
    "_rnn.weight_ih_l2": "module.1.lstm.weight_ih_l2",
    "_rnn.weight_hh_l2": "module.1.lstm.weight_hh_l2",
    "_rnn.bias_ih_l2": "module.1.lstm.bias_ih_l2",
    "_rnn.bias_hh_l2": "module.1.lstm.bias_hh_l2",
    # "_linear.weight": "module.2.module.0.weight",
    # "_linear.bias": "module.2.module.0.bias",
}


def adapt_reinvent_checkpoint(file_path, target_path="/tmp", device=None):
    """Loads a Reinvent pretrained model and make the necessary changes for it to be compatible with pytorchrl"""

    target_path_policy_weights = os.path.join(target_path, "network_policy_params.init")
    target_path_value_weights = os.path.join(target_path, "network_value_params.init")

    if device is None and torch.cuda.is_available():
        save_dict = torch.load(file_path)
    elif device is not None:
        save_dict = torch.load(file_path, map_location=device)
    else:
        save_dict = torch.load(file_path, map_location=lambda storage, loc: storage)

    # Change network weight names
    new_policy_save_dict = {}
    for k in save_dict["network"].keys():
        new_policy_save_dict[reinvent_weights_policy_mapping[k]] = save_dict["network"][k]

    # Change network value names
    new_value_save_dict = {}
    for k in save_dict["network"].keys():
        if k in reinvent_weights_value_mapping:
            new_value_save_dict[reinvent_weights_value_mapping[k]] = save_dict["network"][k]

    # Temporarily save network weights to /tmp
    torch.save(new_policy_save_dict, target_path_policy_weights)
    torch.save(new_value_save_dict, target_path_value_weights)

    # Remove unnecessary network parameters
    network_params = save_dict["network_params"]
    network_params.pop("cell_type", None)
    network_params["embedding_size"] = network_params.pop("embedding_layer_size", None)

    return (
        DeNovoVocabulary(save_dict["vocabulary"], save_dict["tokenizer"]),
        save_dict["max_sequence_length"],
        save_dict["network_params"],
        target_path_policy_weights,
        target_path_value_weights,
    )


def partially_load_checkpoint(module, submodule_name, checkpoint, map_location=None):
    """Load `submodule_name` to `module` from checkpoint."""
    current_state = module.state_dict()
    checkpoint_state = torch.load(checkpoint, map_location=map_location)
    for name, param in checkpoint_state.items():
        if name.startswith(submodule_name):
            current_state[name].copy_(param)
