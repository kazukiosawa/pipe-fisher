from transformers import OPTModel, OPTConfig
from transformers import T5Model, T5Config


# prepare a minimum size dummy OPT model
dummy_config = OPTConfig.from_dict({
    'vocab_size': 1,
    'hidden_size': 1,
    'num_hidden_layers': 1,
    'ffn_dim': 1,
    'max_position_embeddings': 1,
    'num_attention_heads': 1,
    'pad_token_id': 0,
})
dummy_model = OPTModel(dummy_config)
OPTDecoderLayer = dummy_model.decoder.layers[0].__class__


# prepare a minimum size dummy T5 model
dummy_config = T5Config.from_dict({
    'vocab_size': 1,
    'd_model': 1,
    'd_kv': 1,
    'd_ff': 1,
    'num_layers': 1,
    'num_heads': 1,
})
dummy_model = T5Model(dummy_config)
T5Block = dummy_model.encoder.block[0].__class__
