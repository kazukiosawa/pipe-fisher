import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers.models.bert import BertForPreTraining, BertConfig, BertTokenizer
from bert_model import get_stage_bert_for_pretraining
from bert_dataset import BERTDataset

corpus_path = 'bert_data/wikipedia.segmented.nltk.txt'
vocab_file = 'bert_data/bert-large-uncased-vocab.txt'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
batch_size = 4
max_seq_len = 32

tokenizer = BertTokenizer(vocab_file=vocab_file)
dataset = BERTDataset(corpus_path, tokenizer, max_seq_len, corpus_lines=100)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

input_source = next(iter(dataloader))
for key, value in input_source.items():
    input_source[key] = value.to(device)

config = BertConfig.from_dict({
    'hidden_size': 128,
    'num_attention_heads': 2,
    'num_hidden_layers': 6,
    'vocab_size': tokenizer.vocab_size,
    'intermediate_size': 32,
    'max_position_embeddings': 32,
})

model = BertForPreTraining(config)
model.to(device)
for name, module in model.named_modules():
    if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
        print(name)

outputs = model(**input_source)
print('-------')
print('outputs')
for key, value in outputs.items():
    print(key, value.shape)

print('====================')

num_stages = 5
inputs = {}
outputs = None
for stage_id in range(num_stages):
    print('stage', stage_id)
    stage_module = get_stage_bert_for_pretraining(stage_id=stage_id,
                                                  num_stages=num_stages,
                                                  config=config)
    stage_module.to(device)
    for name, module in stage_module.named_modules():
        if isinstance(module, (nn.Linear, nn.Embedding, nn.LayerNorm)):
            print(name)
    inputs = {key: input_source[key] for key in stage_module.keys_from_source}
    if stage_id > 0:
        for key, _ in stage_module.keys_and_sizes_from_prev_stage:
            inputs[key] = outputs[key]
    print('-------')
    print('inputs')
    for key, value in inputs.items():
        print(key, value.shape)
    outputs = stage_module(**inputs)
    print('-------')
    print('outputs')
    for key, value in outputs.items():
        print(key, value.shape)
    print('--------------')

