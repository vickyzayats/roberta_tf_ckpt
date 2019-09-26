## Transfering RoBERTa checkpoint to Tensorflow
This is short explanation and a code about how to move PyTorch checkpoint for RoBERTa to work on Tensorflow.

I use the RoBERTa model from [HuggingFace repository](https://github.com/huggingface/transformers), and adapt some of their scripts. You need to install HuggingFace models in order to use this script. But it's simple, e.g.:
```
pip install transformers
```
You need to modify your Tensorflow config file to include updated vocabulary size, and **add a flag for RoBERTa** (take a look at config.json for roberta-large model). RoBERTa vocabulary size is 50265, but this does not account for any special tokens you might want to include during finetuning. If you want to include any additional vocabulary, just modify this number (e.g. 50275 for additional 100 special tokens).

Now, in order to transfer the checkpoint, run:
```
python3 convert_pytorch_checkpoint_to_tf.py --model_name=roberta-large --config_file=/path/to/config/file --tf_cache_dir=/path/to/output/dir
```
As a side note, HuggingFace release contains multiple RoBERTa models (e.g. roberta-large, roberta-base, etc.). 

Also, you need to update Tensorflow BERT model, bert/modeling.py, with small changes. This repo contains the updated modeling.py file.

## RoBERTa tokenizaiton

I am using HuggingFace tokenization in order to match RoBERTa. Here is an example:

```
from pytorch_transformers import RobertaTokenizer as tokenization

text = 'Here is an example sentence'

tokenizer = tokenization.from_pretrained('roberta-large')
tokens = [tokenizer.tokenize(token) for token in text.split(' ')]
input_ids = tokenizer.convert_tokens_to_ids(tokens)
```

To add additional special tokens:
```
special_tokens = ['[Q]','[Paragraph]','[Table]']
tokenizer.add_tokens(special_tokens)
```

Now you can use it with the original BERT model release. That's easy! 

#### On a side note

If you want to use the original FAIR model, you can use [their script](https://github.com/huggingface/transformers/blob/master/transformers/convert_roberta_original_pytorch_checkpoint_to_pytorch.py) to transfer the model to HuggingFace framework first, and then transfer it to Tensorflow. But I haven't tried it myself.

