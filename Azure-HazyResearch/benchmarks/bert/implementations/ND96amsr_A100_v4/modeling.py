# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018-2022, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

import copy
import json
import logging
import math
import os
import sys
from io import open
from operator import mul
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.utils import checkpoint

from utils import get_rank

from einops import rearrange

from apex.contrib.layer_norm import FastLayerNorm

from model.layers.activations import ACT2FN

from model.layers.embeddings import BertEmbeddings

from model.ops.bert_padding import unpad_input, pad_input, index_first_axis, index_first_axis_residual

from model.losses.cross_entropy_apex import CrossEntropyLossApex
from model.ops.fused_dense import fused_dense_function_td, fused_dense_residual_function
from model.ops.fused_dense import FusedDenseTD, FusedDenseResidual, FusedDenseResGeluDense
from model.ops.layer_norm import dropout_add_layer_norm

# from flash_attn.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func
from model.ops.flash_attn_interface import flash_attn_unpadded_qkvpacked_func, flash_attn_unpadded_kvpacked_func
from model.ops.flash_attn_interface import flash_attn_unpadded_qkvpacked_split_func


logger = logging.getLogger(__name__)


def remap_attn_names_tf(name):
    if 'attention' in name:
        ind = name.index("attention")
        if 'self' in name and 'query' in name and 'kernel' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'q_weight']
        if 'self' in name and 'query' in name and 'bias' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'q_bias']
        if 'self' in name and 'key' in name and 'kernel' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'k_weight']
        if 'self' in name and 'key' in name and 'bias' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'k_bias']
        if 'self' in name and 'value' in name and 'kernel' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'v_weight']
        if 'self' in name and 'value' in name and 'bias' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'v_bias']
        if 'output' in name and 'dense' in name and 'kernel' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'out_proj_weight']
        if 'output' in name and 'dense' in name and 'bias' in name:
            name = name[:(ind+1)] + ['multi_head_attention', 'out_proj_bias']
        if 'output' in name and 'LayerNorm' in name:
            name = name[:(ind+1)] + ['layer_norm'] + name[-1:]
    return name


def load_tf_weights_in_bert(model, tf_checkpoint_path, use_fast_mha=False):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    if get_rank() == 0:
        print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        if get_rank() == 0:
            print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    # MHA params need to be treated separately
    if use_fast_mha:
        mha_params = ['q_weight', 'q_bias', 'k_weight', 'k_bias', 'v_weight', 'v_bias', 'out_proj_weight', 'out_proj_bias']
    else:
        mha_params = []

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step", "LAMB", "LAMB_1", "beta1_power", "beta2_power"] for n in name):
            if get_rank() == 0:
                print("Skipping {}".format("/".join(name)))
            continue

        if use_fast_mha:
            name = remap_attn_names_tf(name)

        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] in mha_params:
                pointer = getattr(pointer, l[0])
            elif l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel' or (m_name in mha_params and 'bias' not in m_name):
            array = np.ascontiguousarray(np.transpose(array))

        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            # If copying smaller into larger, assume padded and ok
            if reduce(mul, pointer.shape) > reduce(mul, array.shape):
                if get_rank() == 0:
                    print("Initialize padded PyTorch weight {}".format(name))
                pointer.data.zero_()

                def generate_slices():
                    slices = []
                    for i in range(array.ndim):
                        slices.append(slice(0, array.shape[i], 1))
                    return slices
                # pointer.data[generate_slices()] = torch.from_numpy(array)
                pointer.data[generate_slices()] = torch.from_numpy(array)
            else:
                e.args += (pointer.shape, array.shape)
                raise
        else:
            if get_rank() == 0:
                print("Initialize PyTorch weight {}".format(name))
            pointer.data = torch.from_numpy(array)
    return model


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertFlashSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.p_dropout = config.attention_probs_dropout_prob
        self.fuse_bias = getattr(config, 'fused_bias_mha', False)

        # linear_cls = nn.Linear if not self.fuse_bias else FusedDenseTD
        linear_cls = nn.Linear if not self.fuse_bias else FusedDenseResidual
        self.Wqkv = linear_cls(self.all_head_size, 3 * config.hidden_size)

    def forward(self, hidden_states, cu_seqlens, max_seqlen_in_batch,
                subset_idx=None, subset_cu_seqlens=None, max_seqlen1=None, batch_size0=None):
        """
        Arguments:
            hidden_states: (total_nnz, dim)
            cu_seqlens: (batch + 1,), torch.int32
            max_seqlen_in_batch: int
        Return:
            out: (total_nnz, dim)
        """
        if subset_idx is None:
            if not self.fuse_bias:
                qkv = self.Wqkv(hidden_states)
            else:
                qkv, hidden_states = self.Wqkv(hidden_states)  # (total_nnz, 3 * dim)
            qkv = rearrange(qkv, 'nnz (t h d) -> nnz t h d', t=3, h=self.num_attention_heads)
            if max_seqlen1 is None or batch_size0 is None:
                out = flash_attn_unpadded_qkvpacked_func(qkv, cu_seqlens, max_seqlen_in_batch,
                                                        self.p_dropout if self.training else 0.0)
            else:
                out = flash_attn_unpadded_qkvpacked_split_func(
                    qkv, cu_seqlens, max_seqlen_in_batch, max_seqlen1, batch_size0,
                    self.p_dropout if self.training else 0.0)
            return rearrange(out, 'nnz h d -> nnz (h d)'), hidden_states
        else:
            hidden_states_subset, hidden_states = index_first_axis_residual(hidden_states, subset_idx)
            dim = hidden_states.shape[-1]
            Wq, Wkv = self.Wqkv.weight[:dim], self.Wqkv.weight[dim:]
            bq, bkv = self.Wqkv.bias[:dim], self.Wqkv.bias[dim:]
            if not self.fuse_bias:
                q = F.linear(hidden_states_subset, Wq, bq)
                kv = F.linear(hidden_states, Wkv, bkv)
            else:
                q, hidden_states_subset = fused_dense_residual_function(hidden_states_subset, Wq, bq)
                kv = fused_dense_function_td(hidden_states, Wkv, bkv)
            q = rearrange(q, 'nnz (h d) -> nnz h d', h=self.num_attention_heads)
            kv = rearrange(kv, 'nnz (t h d) -> nnz t h d', t=2, h=self.num_attention_heads)
            # It's ok to set max_seqlen_q to be much larger
            max_seqlen_subset = max_seqlen_in_batch
            out = flash_attn_unpadded_kvpacked_func(
                q, kv, subset_cu_seqlens, cu_seqlens, max_seqlen_subset, max_seqlen_in_batch,
                self.p_dropout if self.training else 0.0
            )
            return rearrange(out, 'nnz h d -> nnz (h d)'), hidden_states_subset


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fuse_bias = getattr(config, 'fused_bias_mha', False)
        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        linear_cls = nn.Linear if not self.fuse_bias else FusedDenseTD
        self.dense = linear_cls(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        if not self.fused_dropout_add_ln:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        else:
            hidden_states = dropout_add_layer_norm(hidden_states, input_tensor,
                                                   self.LayerNorm.weight, self.LayerNorm.bias,
                                                   self.dropout.p if self.training else 0.0,
                                                   self.LayerNorm.eps)
        return hidden_states


class BertFlashAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.self = BertFlashSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, cu_seqlens, max_s, subset_idx=None, subset_cu_seqlens=None,
                max_seqlen1=None, batch_size0=None):
        """subset_idx: set of indices whose values we care about at the end of the layer
        (e.g., the masked tokens, if this is the final layer).
        """
        self_output, input_tensor = self.self(input_tensor, cu_seqlens, max_s,
                                              subset_idx, subset_cu_seqlens,
                                              max_seqlen1=max_seqlen1, batch_size0=batch_size0)
        return self.output(self_output, input_tensor)


class BertFusedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fused_dropout_add_ln = getattr(config, 'fused_dropout_add_ln', False)
        self.dense_gelu_dense = FusedDenseResGeluDense(config.hidden_size, config.intermediate_size,
                                                       config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_tensor):
        hidden_states, input_tensor = self.dense_gelu_dense(input_tensor)
        if not self.fused_dropout_add_ln:
            hidden_states = self.LayerNorm(self.dropout(hidden_states) + input_tensor)
        else:
            hidden_states = dropout_add_layer_norm(hidden_states, input_tensor,
                                                   self.LayerNorm.weight, self.LayerNorm.bias,
                                                   self.dropout.p if self.training else 0.0,
                                                   self.LayerNorm.eps)
        return hidden_states


class BertLayerUnpad(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertFlashAttention(config)
        self.fused_dense_gelu_dense = getattr(config, 'fused_dense_gelu_dense', False)
        assert self.fused_dense_gelu_dense
        self.fused_mlp = BertFusedMLP(config)

    def forward(self, hidden_states, cu_seqlens, max_seqlen, subset_idx=None, subset_cu_seqlens=None,
                max_seqlen1=None, batch_size0=None):
        """subset_idx: set of indices whose values we care about at the end of the layer
        (e.g., the masked tokens, if this is the final layer).
        """
        attention_output = self.attention(hidden_states, cu_seqlens, max_seqlen,
                                          subset_idx, subset_cu_seqlens,
                                          max_seqlen1=max_seqlen1, batch_size0=batch_size0)
        layer_output = self.fused_mlp(attention_output)
        return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayerUnpad(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])
        assert config.unpad
        self.attn_split = config.attn_split

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True,
                subset_mask=None):
        assert not output_all_encoded_layers
        all_encoder_layers = []
        # attention_mask_bool = rearrange(attention_mask, 'b 1 1 s -> b s') == 0.0
        attention_mask_bool = attention_mask.bool()
        batch, seqlen = hidden_states.shape[:2]
        if not self.attn_split:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch = unpad_input(
                hidden_states, attention_mask_bool
            )
            max_seqlen1, batch_size0 = None, None
        else:
            hidden_states, indices, cu_seqlens, max_seqlen_in_batch, max_seqlen1, batch_size0 = unpad_input(
                hidden_states, attention_mask_bool, max_seqlen1=128
            )
        if subset_mask is None:
            for layer_module in self.layer:
                hidden_states = layer_module(hidden_states, cu_seqlens, max_seqlen_in_batch,
                                             max_seqlen1=max_seqlen1, batch_size0=batch_size0)
            hidden_states = pad_input(hidden_states, indices, batch, seqlen)
        else:
            for layer_module in self.layer[:-1]:
                hidden_states = layer_module(hidden_states, cu_seqlens, max_seqlen_in_batch,
                                             max_seqlen1=max_seqlen1, batch_size0=batch_size0)
            subset_idx = torch.nonzero(subset_mask[attention_mask_bool], as_tuple=False).flatten()
            subset_seqlens = (subset_mask & attention_mask_bool).sum(dim=-1, dtype=torch.int32)
            subset_cu_seqlens = F.pad(torch.cumsum(subset_seqlens, dim=0, dtype=torch.torch.int32),
                                        (1, 0))
            hidden_states = self.layer[-1](hidden_states, cu_seqlens, max_seqlen_in_batch,
                                            subset_idx=subset_idx,
                                            subset_cu_seqlens=subset_cu_seqlens)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states, pool=True):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0] if pool else hidden_states
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = FastLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.fused_fc = getattr(config, 'fused_bias_fc_loss_head', False)
        self.transform = BertPredictionHeadTransform(config)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))
        self.decoder.weight = bert_model_embedding_weights

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        if not self.fused_fc:
            hidden_states = self.decoder(hidden_states) + self.bias
        else:
            hidden_states = fused_dense_function_td(hidden_states, self.decoder.weight, self.bias)
        return hidden_states


class BertPreTrainingHeads(nn.Module):

    def __init__(self, config, bert_model_embedding_weights):
        super().__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPreTrainedModel(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

        # we want to make sure vocab size is padded to % 8 == 0
        if self.config.vocab_size % 8 != 0:
            self.config.vocab_size += 8 - (self.config.vocab_size % 8)
            if get_rank == 0:
                print(f'Padded vocab_size to : {self.config.vocab_size}')

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_checkpoint, state_dict=None, cache_dir=None,
                        from_tf=False, config=None, *inputs, **kwargs):
        """
        Instantiate a BertPreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a BertForPretraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for BertForSequenceClassification)
        """
        logger.info("loading archive file {}".format(pretrained_checkpoint))
        assert config, "BERT configuration file must be provided to from_pretraining()"
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            state_dict = torch.load(pretrained_checkpoint, map_location='cpu' if not torch.cuda.is_available() else None)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return load_tf_weights_in_bert(model, pretrained_checkpoint, use_fast_mha=config.fused_mha)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        # print(f'loading keys: {state_dict.keys()}')
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model


class BertModel(BertPreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.BertModel(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)
        self.unpad = config.unpad
        assert self.unpad

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, position_ids=None,
                output_all_encoded_layers=True, masked_tokens_mask=None):
        batch_size, seq_length = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask#.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        if self.unpad == False:
            extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        embedding_output = self.embeddings(input_ids, token_type_ids, position_ids)

        if masked_tokens_mask is not None:
            # We also need the first column for the CLS token
            first_col_mask = torch.zeros(batch_size, seq_length, dtype=torch.bool, device=device)
            first_col_mask[:, 0] = True
            subset_mask = masked_tokens_mask | first_col_mask
        else:
            subset_mask = None

        encoder_outputs = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,
                                      subset_mask=subset_mask)

        if masked_tokens_mask is None:
            sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        else:
            # TD [2022-03-01]: the indexing here is very tricky.
            attention_mask_bool = attention_mask > 0
            subset_idx = subset_mask[attention_mask_bool]
            sequence_output = encoder_outputs[0][masked_tokens_mask[attention_mask_bool][subset_idx]]
            pool_input = encoder_outputs[0][first_col_mask[attention_mask_bool][subset_idx]]
            pooled_output = (self.pooler(pool_input, pool=False)
                             if self.pooler is not None else None)

        if not output_all_encoded_layers:
            encoder_outputs = sequence_output
        return encoder_outputs, pooled_output


class BertForPretraining(BertPreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = BertForPretraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertForPretraining, self).__init__(config)
        self.bert = BertModel(config)
        self.cls = BertPreTrainingHeads(config, self.bert.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)
        self.dense_seq_output = config.dense_seq_output
        # If last_layer_subset, we only need the compute the last layer for a subset of tokens
        # (e.g., the tokens we need to compute the masked LM loss and the next-sentence prediction).
        self.last_layer_subset = getattr(config, 'last_layer_subset', False)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
                masked_lm_labels=None, next_sentence_label=None):
        masked_tokens_mask = masked_lm_labels > 0 if (self.last_layer_subset and masked_lm_labels is not None) else None
        sequence_output, pooled_output = self.bert(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False,
                                                   masked_tokens_mask=masked_tokens_mask)
        if self.dense_seq_output and masked_lm_labels is not None:
            masked_token_idx = torch.nonzero(masked_lm_labels.flatten() > 0, as_tuple=False).flatten()
            if not self.last_layer_subset:
                sequence_output = index_first_axis(rearrange(sequence_output, 'b s d -> (b s) d'),
                                                   masked_token_idx)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLossApex(ignore_index=0)
            if masked_token_idx is not None:  # prediction_scores are already flattened
                masked_lm_loss = loss_fct(prediction_scores, masked_lm_labels.flatten()[masked_token_idx])
            else:
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            nsp_loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = nsp_loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            #print("loss is {} {}".format(masked_lm_loss, next_sentence_loss))
            total_loss = (masked_lm_loss + next_sentence_loss).float()

            masked_lm_labels_flat = masked_lm_labels.view(-1)

            # Masked Language Model Accuracy
            mlm_labels = masked_lm_labels_flat[masked_lm_labels_flat != 0]
            if not self.dense_seq_output:
                prediction_scores_flat = prediction_scores.view(-1, prediction_scores.shape[-1])
                mlm_predictions_scores = prediction_scores_flat[masked_lm_labels_flat != 0]
                mlm_predictions = mlm_predictions_scores.argmax(dim=-1)
            else:
                mlm_predictions = prediction_scores.argmax(dim=-1)

            mlm_acc = (mlm_predictions == mlm_labels).sum(dtype=torch.float) / mlm_labels.numel()

            # We need mlm_labels.numel() as torch.Tensor to be sent via DDP
            return total_loss, mlm_acc, torch.tensor(mlm_labels.numel(), device=mlm_acc.device)
        else: #TODO: Handle this path for dense sequence output as well
            return prediction_scores, seq_relationship_score
