import mxnet as mx
import torch
from transformers import BertForMaskedLM

from mlm.src.mlm.models import get_pretrained
from mlm.src.mlm.scorers import MLMScorerPT
from tokenizer.dictatokenizer import DictaAutoTokenizer

ctxs = [mx.cpu()] # or, e.g., [mx.gpu(0), mx.gpu(1)]


# model_path = './AlephBertGimmel_62600'
dicta_model_path = './lm-dicta'


real_tokenizer = DictaAutoTokenizer.from_pretrained(dicta_model_path)

# take our input sentences, and encode them [tokenize them into tokens, and then convert them to IDs]
sentence = 'אני רוצה לבדוק את המודל ברט של דיקטה עם ריצ\'רץ\' וגם וגם גמ"חים שונים'
sentence2 = 'שלום לך'
sentence3 = 'ויאמר ה'

# tokenized = real_tokenizer.encode(sentence)
# Convert the input to a torch tensor, and unsqueeze to add the batch dimension
# tokenized_input = torch.tensor(tokenized).unsqueeze(0)

# # for using the encoder model for post-training (using `huggingface` pipeline or external)
# model = BertModel.from_pretrained(model_path)
# output = model(tokenized_input)
# # print(model.eval())
# # print(output)
# print(output.last_hidden_state.shape)  # [1 x total_tokens x hidden_dim]
# print(output.pooler_output.shape)  # [1 x x hidden_dim]

# for using the masked-lm model for token prediction
# model = BertForMaskedLM.from_pretrained(model_path)
# output = model(tokenized_input)
# print(output.logits.shape)  # [1 x total_tokens x vocab_size]
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
model, vocab, _ = get_pretrained(ctxs = ctxs, name="dicta", params_file=dicta_model_path)
scorer = MLMScorerPT(model, vocab, real_tokenizer, ctxs)
print(sentence ,scorer.score_sentences([sentence]))

print(sentence2,scorer.score_sentences([sentence2]))

print(sentence3,scorer.score_sentences([sentence3]))