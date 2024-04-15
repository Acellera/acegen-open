from copy import deepcopy
from typing import Optional

import torch
from tensordict.nn import TensorDictModule, TensorDictSequential
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import ExplorationType
from torchrl.modules import ActorValueOperator, LSTMModule, MLP, ProbabilisticActor


class Embed(torch.nn.Module):
    """Implements a simple embedding layer.

    It handles the case of having a time dimension (RL training) and not having
    it (RL inference).

    Args:
        input_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
    """

    def __init__(self, input_size, embedding_size):
        super().__init__()
        self.input_size = input_size
        self.embedding_size = embedding_size
        self._embedding = torch.nn.Embedding(input_size, embedding_size)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        *batch, L = inputs.shape
        if len(batch) > 1:
            inputs = inputs.flatten(0, len(batch) - 1)
        inputs = inputs.squeeze(-1)  # Embedding creates an extra dimension
        out = self._embedding(inputs)
        if len(batch) > 1:
            out = out.unflatten(0, batch)
        return out


def create_lstm_components(
    vocabulary_size: int,
    embedding_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
    layer_norm: bool = True,
    output_size: Optional[int] = None,
    in_key: str = "observation",
    out_key: str = "logits",
    recurrent_state: str = "recurrent_state",
    python_based: bool = False,
):
    """Create all LSTM model components: embedding, LSTM, and head.

    These modules handle the case of having a time dimension (RL training)
    and not having it (RL inference).

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the LSTM hidden state.
        num_layers (int): The number of LSTM layers.
        dropout (float): The LSTM dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        output_size (int): The size of the output logits.
        in_key (str): The input key name.
        out_key (str): The output key name.
        recurrent_state (str): The name of the recurrent state.
        python_based (bool): Whether to use the Python-based LSTM module.
            Default is False, a CuDNN-based LSTM module is used.

    Example:
    ```python
    training_model, inference_model = create_model(10)
    ```
    """
    embedding_module = TensorDictModule(
        Embed(vocabulary_size, embedding_size),
        in_keys=[in_key],
        out_keys=["embed"],
    )
    lstm_module = LSTMModule(
        dropout=dropout,
        input_size=embedding_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        in_keys=["embed", f"{recurrent_state}_h", f"{recurrent_state}_c", "is_init"],
        out_keys=[
            "features",
            ("next", f"{recurrent_state}_h"),
            ("next", f"{recurrent_state}_c"),
        ],
        python_based=python_based,
    )
    head = TensorDictModule(
        MLP(
            in_features=hidden_size,
            out_features=output_size or vocabulary_size,
            num_cells=[],
            dropout=dropout,
            norm_class=torch.nn.LayerNorm if layer_norm else None,
            norm_kwargs={"normalized_shape": hidden_size} if layer_norm else {},
        ),
        in_keys=["features"],
        out_keys=[out_key],
    )

    return embedding_module, lstm_module, head


def create_lstm_actor(
    vocabulary_size: int,
    embedding_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
    layer_norm: bool = False,
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True,
    in_key: str = "observation",
    out_key: str = "logits",
    recurrent_state: str = "recurrent_state_actor",
    python_based: bool = False,
):
    """Create one LSTM-based actor model for inference and one for training.

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the LSTM hidden state.
        num_layers (int): The number of LSTM layers.
        dropout (float): The LSTM dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        distribution_class (torch.distributions.Distribution): The
            distribution class to use.
        return_log_prob (bool): Whether to return the log probability
            of the action.
        in_key (str): The input key name.
        out_key (str):): The output key name.
        recurrent_state (str): The name of the recurrent state.
        python_based (bool): Whether to use the Python-based LSTM module.
            Default is False, a CuDNN-based LSTM module is used.

    Example:
    ```python
    training_actor, inference_actor = create_lstm_actor(10)
    ```
    """
    embedding, lstm, head = create_lstm_components(
        vocabulary_size,
        embedding_size,
        hidden_size,
        num_layers,
        dropout,
        layer_norm,
        vocabulary_size,
        in_key,
        out_key,
        recurrent_state,
        python_based,
    )
    spec = CompositeSpec(
        **{
            f"{recurrent_state}_h": UnboundedContinuousTensorSpec(
                shape=torch.Size([lstm.lstm.num_layers, lstm.lstm.hidden_size]),
                dtype=torch.float32,
            ),
            f"{recurrent_state}_c": UnboundedContinuousTensorSpec(
                shape=torch.Size([lstm.lstm.num_layers, lstm.lstm.hidden_size]),
                dtype=torch.float32,
            ),
        }
    )
    actor_inference_model = TensorDictSequential(embedding, lstm, head)
    actor_training_model = TensorDictSequential(
        embedding,
        lstm.set_recurrent_mode(True),
        head,
    )

    actor_inference_model = ProbabilisticActor(
        module=actor_inference_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=distribution_class,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )

    actor_training_model = ProbabilisticActor(
        module=actor_training_model,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=distribution_class,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )

    actor_training_model.rnn_spec = spec
    actor_inference_model.rnn_spec = deepcopy(spec)

    return actor_training_model, actor_inference_model


def create_lstm_critic(
    vocabulary_size: int,
    embedding_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 3,
    dropout: float = 0.0,
    layer_norm: bool = False,
    critic_value_per_action=False,
    in_key: str = "observation",
    recurrent_state: str = "recurrent_state_critic",
    python_based: bool = False,
):
    """Create one LSTM-based critic model for inference and one for training.

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the LSTM hidden state.
        num_layers (int): The number of LSTM layers.
        dropout (float): The LSTM dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        critic_value_per_action (bool): Whether the critic should output a
            value per action or a single value.
        in_key (Union[str, List[str]]): The input key name.
        recurrent_state (str): The name of the recurrent state.
        python_based (bool): Whether to use the Python-based LSTM module.
            Default is False, a CuDNN-based LSTM module is used.

    Example:
    ```python
    training_critic, inference_critic = create_lstm_critic(10)
    ```
    """
    output_size = vocabulary_size if critic_value_per_action else 1
    out_key = "action_value" if critic_value_per_action else "state_value"

    embedding, lstm, head = create_lstm_components(
        vocabulary_size,
        embedding_size,
        hidden_size,
        num_layers,
        dropout,
        layer_norm,
        output_size,
        in_key,
        out_key,
        recurrent_state,
        python_based,
    )

    spec = CompositeSpec(
        **{
            f"{recurrent_state}_h": UnboundedContinuousTensorSpec(
                shape=torch.Size([lstm.lstm.num_layers, lstm.lstm.hidden_size]),
                dtype=torch.float32,
            ),
            f"{recurrent_state}_c": UnboundedContinuousTensorSpec(
                shape=torch.Size([lstm.lstm.num_layers, lstm.lstm.hidden_size]),
                dtype=torch.float32,
            ),
        }
    )

    critic_inference_model = TensorDictSequential(embedding, lstm, head)
    critic_training_model = TensorDictSequential(
        embedding, lstm.set_recurrent_mode(True), head
    )
    critic_training_model.rnn_spec = spec
    critic_inference_model.rnn_spec = deepcopy(spec)
    return critic_training_model, critic_inference_model


def create_lstm_actor_critic(
    vocabulary_size: int,
    embedding_size: int = 256,
    hidden_size: int = 512,
    num_layers: int = 3,
    dropout: float = 0.00,
    layer_norm: bool = False,
    distribution_class=torch.distributions.Categorical,
    return_log_prob=True,
    critic_value_per_action=False,
    in_key: str = "observation",
    out_key: str = "logits",
    recurrent_state: str = "recurrent_state",
    python_based: bool = False,
):
    """Create a LSTM-based actor-critic model for inference and one for training.

    Args:
        vocabulary_size (int): The number of possible input values.
        embedding_size (int): The size of the embedding vectors.
        hidden_size (int): The size of the LSTM hidden state.
        num_layers (int): The number of LSTM layers.
        dropout (float): The LSTM dropout rate.
        layer_norm (bool): Whether to use layer normalization.
        layer_norm (bool): Whether to use layer normalization.
        distribution_class (torch.distributions.Distribution): The
            distribution class to use.
        return_log_prob (bool): Whether to return the log probability
            of the action.
        critic_value_per_action (bool): Whether the critic should output
            a value per action or a single value.
        in_key (str): The input key name.
        out_key (str): The output key name.
        recurrent_state (str): The name of the recurrent state.
        python_based (bool): Whether to use the Python-based LSTM module.
            Default is False, a CuDNN-based LSTM module is used.

    Example:
    ```python
    (training_actor, inference_actor, training_critic,
        inference_critic) = create_lstm_actor_critic(10)
    ```
    """
    embedding, lstm, actor_head = create_lstm_components(
        vocabulary_size,
        embedding_size,
        hidden_size,
        num_layers,
        dropout,
        layer_norm,
        vocabulary_size,
        in_key,
        out_key,
        recurrent_state,
        python_based,
    )

    spec = CompositeSpec(
        **{
            f"{recurrent_state}_h": UnboundedContinuousTensorSpec(
                shape=torch.Size([lstm.lstm.num_layers, lstm.lstm.hidden_size]),
                dtype=torch.float32,
            ),
            f"{recurrent_state}_c": UnboundedContinuousTensorSpec(
                shape=torch.Size([lstm.lstm.num_layers, lstm.lstm.hidden_size]),
                dtype=torch.float32,
            ),
        }
    )

    actor_head = ProbabilisticActor(
        module=actor_head,
        in_keys=["logits"],
        out_keys=["action"],
        distribution_class=distribution_class,
        return_log_prob=return_log_prob,
        default_interaction_type=ExplorationType.RANDOM,
    )
    critic_out = ["action_value"] if critic_value_per_action else ["state_value"]
    critic_head = TensorDictModule(
        MLP(
            in_features=hidden_size,
            out_features=vocabulary_size if critic_value_per_action else 1,
            num_cells=[],
        ),
        in_keys=["features"],
        out_keys=critic_out,
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic_inference = ActorValueOperator(
        common_operator=TensorDictSequential(embedding, lstm),
        policy_operator=actor_head,
        value_operator=critic_head,
    )

    common_net = TensorDictSequential(embedding, lstm.set_recurrent_mode(True))
    actor_critic_training = ActorValueOperator(
        common_operator=common_net,
        policy_operator=actor_head,
        value_operator=critic_head,
    )

    actor_inference = actor_critic_inference.get_policy_operator()
    critic_inference = actor_critic_inference.get_value_operator()
    actor_training = actor_critic_training.get_policy_operator()
    critic_training = actor_critic_training.get_value_operator()

    actor_training.rnn_spec = spec
    actor_inference.rnn_spec = deepcopy(spec)
    critic_training.rnn_spec = deepcopy(spec)
    critic_inference.rnn_spec = deepcopy(spec)

    return actor_training, actor_inference, critic_training, critic_inference
