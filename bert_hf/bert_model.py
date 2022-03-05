from typing import List
from collections import OrderedDict
import copy

from torch.nn import CrossEntropyLoss
from transformers.modeling_utils import ModuleUtilsMixin
from transformers.models.bert import BertConfig, BertForPreTraining, BertModel, BertPreTrainedModel
from pipeline import StageModule

# prepare a minimum size dummy model for extracting Module classes
dummy_config = BertConfig.from_dict({
    'hidden_size': 1,
    'num_attention_heads': 1,
    'num_hidden_layers': 1,
    'vocab_size': 1,
    'intermediate_size': 1,
    'max_position_embeddings': 1,
})
dummy_model = BertForPreTraining(dummy_config)
BertEncoder = dummy_model.bert.encoder.__class__
BertPooler = dummy_model.bert.pooler.__class__
BertPreTrainingHeads = dummy_model.cls.__class__


def get_stage_bert_for_pretraining(stage_id: int,
                                   num_stages: int,
                                   config: BertConfig,
                                   hidden_layers_assignments: List[int] = None) -> StageModule:
    """
    start_stage (stage_id == 0): BertEmbeddings + [BertLayer] * N_s
    intermediate_stage (0 < stage_id < num_stages - 1): [BertLayer] * N_i
    end_stage (stage_id == num_stages - 1): [BertLayer] * N_e + BertPreTrainingHeads

    N_s, N_i, N_e: the number of hidden layers (BertLayers) for each stage
    """
    assert num_stages > 1, 'At least 2 stages are required.'
    if hidden_layers_assignments is None:
        """
        Assign the number of hidden layers (BertLayers) so that
        the following are satisfied: 
            N_e <= N_s <= N_i
        """
        hidden_layers_assignments = [0] * num_stages
        for i in range(config.num_hidden_layers):
            hidden_layers_assignments[-((i + 2) % num_stages)] += 1
    assert len(hidden_layers_assignments) == num_stages
    assert stage_id in range(num_stages)
    # overwrite num_hidden_layers with the number for this stage
    config = copy.deepcopy(config)
    config.num_hidden_layers = hidden_layers_assignments[stage_id]

    if stage_id == 0:
        return StartingStage(config)
    elif stage_id == num_stages - 1:
        return EndingStage(config)
    else:
        return IntermediateStage(config)


class StartingStage(BertModel, StageModule):
    def __init__(self, config):
        super().__init__(config, add_pooling_layer=False)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        return OrderedDict(hidden_states=outputs.last_hidden_state)

    @property
    def keys_from_source(self):
        return ['input_ids', 'attention_mask', 'token_type_ids']

    @property
    def keys_and_sizes_from_prev_stage(self):
        return []

    @property
    def keys_and_sizes_of_next_stage(self):
        return [('hidden_states', (self.config.hidden_size,))]


class IntermediateStage(BertPreTrainedModel, StageModule, ModuleUtilsMixin):
    def __init__(self, config):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.post_init()

    def forward(self, hidden_states, attention_mask):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask,
                                                                   hidden_states.size()[:-1],
                                                                   hidden_states.device)
        outputs = self.encoder.forward(
            hidden_states=hidden_states,
            attention_mask=extended_attention_mask,
            return_dict=True
        )
        return OrderedDict(hidden_states=outputs.last_hidden_state)

    @property
    def keys_from_source(self):
        return ['attention_mask']

    @property
    def keys_and_sizes_from_prev_stage(self):
        return [('hidden_states', (self.config.hidden_size,))]

    @property
    def keys_and_sizes_of_next_stage(self):
        return [('hidden_states', (self.config.hidden_size,))]


class EndingStage(BertPreTrainedModel, StageModule, ModuleUtilsMixin):
    def __init__(self, config, loss_reduction='mean'):
        super().__init__(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.cls = BertPreTrainingHeads(config)
        self.post_init()
        self.loss_reduction = loss_reduction

    def forward(self, hidden_states, attention_mask, labels, next_sentence_label):
        extended_attention_mask = self.get_extended_attention_mask(attention_mask,
                                                                   hidden_states.size()[:-1],
                                                                   hidden_states.device)
        encoder_outputs = self.encoder(hidden_states, extended_attention_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        loss_fct = CrossEntropyLoss(reduction=self.loss_reduction)
        masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
        next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
        total_loss = masked_lm_loss + next_sentence_loss
        return OrderedDict(loss=total_loss)

    @property
    def keys_from_source(self):
        return ['attention_mask', 'labels', 'next_sentence_label']

    @property
    def keys_and_sizes_from_prev_stage(self):
        return [('hidden_states', (self.config.hidden_size,))]

    @property
    def keys_and_sizes_of_next_stage(self):
        return []
