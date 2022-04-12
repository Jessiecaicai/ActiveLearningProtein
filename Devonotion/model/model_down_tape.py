from .model_utils import ValuePredictionHead
from .model_utils import SequenceClassificationHead
from .model_utils import SequenceToSequenceClassificationHead
from .model_utils import PairwiseContactPredictionHead
import torch.nn as nn

from .. import esm

# A40
# esm1, esm1_alphabet = esm.pretrained.load_model_and_alphabet("/home/public/chenlei/protein_representation_learning/pretrained_models/esm1b_t33_650M_UR50S.pt")
# V100
# esm1, esm1_alphabet = esm.pretrained.load_model_and_alphabet("/home/guo/data/pretrained_models/esm1b_t33_650M_UR50S.pt")
esm1, esm1_alphabet = esm.pretrained.esm1b_t33_650M_UR50S()

class Pooler(nn.Module):
    """
    Parameter-free poolers to get the sentence embedding
    'cls': [CLS] representation with BERT/RoBERTa's MLP pooler.
    'cls_before_pooler': [CLS] representation without the original MLP pooler.
    'avg': average of the last layers' hidden states at each token.
    'avg_top2': average of the last two layers.
    'avg_first_last': average of the first and the last layers.
    """
    def __init__(self, pooler_type):
        super().__init__()
        self.pooler_type = pooler_type
        assert self.pooler_type in ["cls", "avg"], "unrecognized pooling type %s" % self.pooler_type

    def forward(self, tokens, outputs):
        last_hidden = outputs
        attention_mask = 1 - tokens.eq(esm1_alphabet.padding_idx).type_as(outputs)

        if self.pooler_type in ['cls']:
            return last_hidden[:, 0]
        elif self.pooler_type == "avg":
            return ((last_hidden * attention_mask.unsqueeze(-1)).sum(1) / attention_mask.sum(-1).unsqueeze(-1))
        else:
            raise NotImplementedError

# task_model('fluorescence', 'transformer')
# task_model('stability', 'transformer')
class ProteinBertForValuePrediction(nn.Module):

    def __init__(self):
        super().__init__()

        self.bert = esm1
        self.predict = ValuePredictionHead(esm1.args.embed_dim)


    def forward(self, input_ids, targets=None, finetune=True, finetune_emb=True):
        pooler_type = "cls"
        pooler = Pooler(pooler_type)

        for k, v in self.bert.named_parameters():
            if not finetune:
                v.requires_grad = False
            elif not finetune_emb and 'embed_tokens.weight' in k:
                v.requires_grad = False
            elif not finetune_emb and 'embed_positions.weight' in k:
                v.requires_grad = False

        outputs = self.bert(input_ids, repr_layers=[33])

        sequence_output = outputs['representations'][33]

        pooled_output = pooler(input_ids, sequence_output)

        outputs = self.predict(pooled_output, targets)
        # (loss), prediction_scores

        return outputs
