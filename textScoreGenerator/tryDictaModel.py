import mxnet as mx
import torch
from transformers import BertForMaskedLM

from mlm.src.mlm.models import get_pretrained
from mlm.src.mlm.scorers import MLMScorerPT
from tokenizer.dictatokenizer import DictaAutoTokenizer

ctxs = [mx.cpu()] # or, e.g., [mx.gpu(0), mx.gpu(1)]


# model_path = './AlephBertGimmel_62600'
model_path = './lm-dicta'


tokenizer = DictaAutoTokenizer.from_pretrained(model_path)

# take our input sentences, and encode them [tokenize them into tokens, and then convert them to IDs]
sentence = 'אני רוצה לבדוק את המודל ברט של דיקטה עם ריצ\'רץ\' וגם וגם גמ"חים שונים'
tokenized = tokenizer.encode(sentence)
# Convert the input to a torch tensor, and unsqueeze to add the batch dimension
tokenized_input = torch.tensor(tokenized).unsqueeze(0)

# # for using the encoder model for post-training (using `huggingface` pipeline or external)
# model = BertModel.from_pretrained(model_path)
# output = model(tokenized_input)
# # print(model.eval())
# # print(output)
# print(output.last_hidden_state.shape)  # [1 x total_tokens x hidden_dim]
# print(output.pooler_output.shape)  # [1 x x hidden_dim]

# for using the masked-lm model for token prediction
model = BertForMaskedLM.from_pretrained(model_path)
output = model(tokenized_input)
print(output.logits.shape)  # [1 x total_tokens x vocab_size]
# vocab = None
#
# # model, vocab, tokenizer = get_pretrained(ctxs, model_path+'/')
# scorer = MLMScorer(model, vocab, tokenizer, ctxs)
# print(scorer.score_sentences([sentence]))
#
# model, vocab, tokenizer = get_pretrained(ctxs, 'bert-base-cased')
# scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
# print(scorer.score_sentences(["Hello world!"]))

# EXPERIMENTAL: PyTorch MLMs (use names from https://huggingface.co/transformers/pretrained_models.html)
model, vocab, tokenizer = get_pretrained(ctxs, 'onlplab/alephbert-base')
scorer = MLMScorerPT(model, vocab, tokenizer, ctxs)
print(scorer.score_sentences([sentence]))