from fairseq.models.roberta import RobertaModel as FairseqRobertaModel
from fairseq.modules import TransformerSentenceEncoderLayer

from transformers.modeling_bert import (BertConfig, BertEncoder,
                                        BertIntermediate, BertLayer,
                                        BertModel, BertOutput,
                                        BertSelfAttention,
                                        BertSelfOutput)
from transformers.modeling_roberta import (RobertaEmbeddings,
                                           RobertaForMaskedLM,
                                           RobertaForSequenceClassification,
                                           RobertaModel)




# roberta_checkpoint_path = 'checkpoints/uniref50'
# data_dir = 'data/uniref50'
# pytorch_dump_folder_path = 'checkpoints/roberta_TF_dump'
# classification_head = False
# roberta = FairseqRobertaModel.from_pretrained(roberta_checkpoint_path, 'checkpoint_best.pt', data_dir)
