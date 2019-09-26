# coding=utf-8
# Copyright 2018 The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert Huggingface Pytorch checkpoint to Tensorflow checkpoint."""

import os
import argparse
import json
import torch
import numpy as np
import tensorflow as tf
tf.enable_eager_execution()
#from pytorch_transformers.modeling_bert import BertModel
from bert.modeling import create_initializer
from pytorch_transformers.modeling_roberta import RobertaModel as BertModel


def convert_pytorch_checkpoint_to_tf(model:BertModel, ckpt_dir:str, model_name:str, config:dict):

    """
    :param model:BertModel Pytorch model instance to be converted
    :param ckpt_dir: Tensorflow model directory
    :param model_name: model name
    :return:

    Currently supported HF models:
        Y BertModel
        N BertForMaskedLM
        N BertForPreTraining
        N BertForMultipleChoice
        N BertForNextSentencePrediction
        N BertForSequenceClassification
        N BertForQuestionAnswering
    """

    tensors_to_transpose = (
        "dense.weight",
        "attention.self.query",
        "attention.self.key",
        "attention.self.value"
    )

    var_map = (
        ('layer.', 'layer_'),
        ('word_embeddings.weight', 'word_embeddings'),
        ('position_embeddings.weight', 'position_embeddings'),
        ('token_type_embeddings.weight', 'token_type_embeddings'),
        ('.', '/'),
        ('LayerNorm/weight', 'LayerNorm/gamma'),
        ('LayerNorm/bias', 'LayerNorm/beta'),
        ('weight', 'kernel')
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    state_dict = model.state_dict()

    def to_tf_var_name(name:str):
        for patt, repl in iter(var_map):
            name = name.replace(patt, repl)
        return 'bert/{}'.format(name)

    def create_tf_var(tensor:np.ndarray, name:str, session:tf.Session):
        tf_dtype = tf.dtypes.as_dtype(tensor.dtype)
        tf_var = tf.get_variable(dtype=tf_dtype, shape=tensor.shape, name=name, initializer=tf.zeros_initializer())
        session.run(tf.variables_initializer([tf_var]))
        session.run(tf_var)
        return tf_var

    tf.reset_default_graph()
    with tf.Session() as session:
        for var_name in state_dict:
            tf_name = to_tf_var_name(var_name)
            torch_tensor = state_dict[var_name].numpy()
            if 'token_type_embeddings' in tf_name:
                torch_tensor = np.tile(torch_tensor,[config['type_vocab_size'],1])
            if 'word_embeddings' in tf_name:
                add_emb_shape = config['vocab_size'] - torch_tensor.shape[0]
                embedding_table = tf.get_variable(name='additional_emb', 
                        shape=[add_emb_shape, torch_tensor.shape[1]], 
                        initializer=create_initializer(config['initializer_range']))
                embedding_table.initializer.run()
                additional_emb = embedding_table.eval()
                torch_tensor = np.concatenate([torch_tensor, additional_emb], axis=0)
                
            if any([x in var_name for x in tensors_to_transpose]):
                torch_tensor = torch_tensor.T
            tf_var = create_tf_var(tensor=torch_tensor, name=tf_name, session=session)
            tf.keras.backend.set_value(tf_var, torch_tensor)
            tf_weight = session.run(tf_var)
            print("Successfully created {}: {}".format(tf_name, np.allclose(tf_weight, torch_tensor)))

        saver = tf.train.Saver(tf.trainable_variables())
        saver.save(session, os.path.join(ckpt_dir, model_name.replace("-", "_") + ".ckpt"))


def main(raw_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",
                        type=str,
                        required=True,
                        help="model name e.g. bert-base-uncased")
    parser.add_argument("--config_file",
                        type=str,
                        required=True,
                        help="config for Tensorflow model")
    parser.add_argument("--cache_dir",
                        type=str,
                        default=None,
                        required=False,
                        help="Directory containing pytorch model")
    parser.add_argument("--tf_cache_dir",
                        type=str,
                        required=True,
                        help="Directory in which to save tensorflow model")
    args = parser.parse_args(raw_args)
   
    with open(args.config_file, 'r') as inf:
      config = json.load(inf)

    if args.cache_dir:
      model = BertModel.from_pretrained(
          pretrained_model_name_or_path=args.model_name,
          cache_dir=args.cache_dir
      )
    else:
      model = BertModel.from_pretrained(
          pretrained_model_name_or_path=args.model_name,
      )
    
    convert_pytorch_checkpoint_to_tf(
        model=model,
        ckpt_dir=args.tf_cache_dir,
        model_name=args.model_name,
        config=config,
    )


if __name__ == "__main__":
    main()
