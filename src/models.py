# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from torch import nn

from .utils import load_embeddings, normalize_embeddings


class Discriminator(nn.Module):

    def __init__(self, params):
        super(Discriminator, self).__init__()

        self.emb_dim = params.emb_dim
        self.dis_layers = params.dis_layers
        self.dis_hid_dim = params.dis_hid_dim
        self.dis_dropout = params.dis_dropout
        self.dis_input_dropout = params.dis_input_dropout

        self.hid_channels = 32

        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=self.hid_channels, kernel_size=7, padding=3),
            nn.ReLU(),
            )
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=self.hid_channels, out_channels=self.hid_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            )
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=self.hid_channels, out_channels=1, kernel_size=3, padding=1),
            nn.Sigmoid(),
            )

    def forward(self, x):
        assert x.dim() == 2 and x.size(1) == self.emb_dim
        x = self.conv1(x.view(-1, 1, self.emb_dim))
        x = self.conv2(x)
        x = self.conv3(x)
        return x.view(-1)


def build_model(params, with_dis):
    """
    Build all components of the model.
    """
    # source embeddings
    src_dico, _src_emb = load_embeddings(params, source=True)
    params.src_dico = src_dico
    src_emb = nn.Embedding(len(src_dico), params.emb_dim, sparse=True)
    src_emb.weight.data.copy_(_src_emb)

    # target embeddings
    if params.tgt_lang:
        tgt_dico, _tgt_emb = load_embeddings(params, source=False)
        params.tgt_dico = tgt_dico
        tgt_emb = nn.Embedding(len(tgt_dico), params.emb_dim, sparse=True)
        tgt_emb.weight.data.copy_(_tgt_emb)
    else:
        tgt_emb = None

    # mapping
    mapping = nn.Linear(params.emb_dim, params.emb_dim, bias=True)
    if getattr(params, 'map_id_init', True):
        mapping.weight.data.copy_(torch.diag(torch.ones(params.emb_dim)))

    # discriminator
    discriminator = Discriminator(params) if with_dis else None

    # cuda
    if params.cuda:
        src_emb.cuda()
        if params.tgt_lang:
            tgt_emb.cuda()
        mapping.cuda()
        if with_dis:
            discriminator.cuda()

    # normalize embeddings
    params.src_mean = normalize_embeddings(src_emb.weight.data, params.normalize_embeddings)
    if params.tgt_lang:
        params.tgt_mean = normalize_embeddings(tgt_emb.weight.data, params.normalize_embeddings)

    return src_emb, tgt_emb, mapping, discriminator
