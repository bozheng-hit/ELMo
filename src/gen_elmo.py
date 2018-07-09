#!/usr/bin/env python
from __future__ import print_function
from __future__ import unicode_literals
import os
import errno
import sys
import codecs
import argparse
import time
import random
import logging
import json
import copy
import tempfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from modules.elmo import ElmobiLm
from modules.lstm import LstmbiLm
from modules.token_embedder import ConvTokenEmbedder, LstmTokenEmbedder
from modules.embedding_layer import EmbeddingLayer
from dataloader import load_embedding
import subprocess
import numpy as np
import h5py
import collections

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s: %(message)s')


def dict2namedtuple(dic):
  return collections.namedtuple('Namespace', dic.keys())(**dic)


def read_corpus(path, max_chars = None):
  """
  read raw text file
  :param path:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for line in fin.read().strip().split('\n'):
      data = ['<bos>']
      text = []
      for token in line.split('\t'):
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def read_conll_corpus(path, max_chars=None):
  """

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        text.append(token)
        if max_chars is not None and len(token) + 2 > max_chars:
          token = token[:max_chars - 2]
        data.append(token)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def read_conll_char_corpus(path, max_chars=None):
  """

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        for ch in token:
          text.append(ch)
          if max_chars is not None and len(ch) + 2 > max_chars:
            ch = ch[:max_chars - 2]
          data.append(ch)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def read_conll_char_vi_corpus(path, max_chars=None):
  """

  :param path:
  :param max_chars:
  :return:
  """
  dataset = []
  textset = []
  with codecs.open(path, 'r', encoding='utf-8') as fin:
    for payload in fin.read().strip().split('\n\n'):
      data = ['<bos>']
      text = []
      lines = payload.splitlines()
      body = [line for line in lines if not line.startswith('#')]
      for line in body:
        fields = line.split('\t')
        num, token = fields[0], fields[1]
        if '-' in num or '.' in num:
          continue
        for ch in token.split():
          text.append(ch)
          if max_chars is not None and len(ch) + 2 > max_chars:
            ch = ch[:max_chars - 2]
          data.append(ch)
      data.append('<eos>')
      dataset.append(data)
      textset.append(text)
  return dataset, textset


def create_one_batch(x, word2id, char2id, config, oov='<oov>', pad='<pad>', sort=True, use_cuda=False):
  batch_size = len(x)
  lst = list(range(batch_size))
  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  lens = [len(x[i]) for i in lst]
  max_len = max(lens)

  if word2id is not None:
    oov_id, pad_id = word2id.get(oov, None), word2id.get(pad, None)
    assert oov_id is not None and pad_id is not None
    batch_w = torch.LongTensor(batch_size, max_len).fill_(pad_id)
    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_w[i][j] = word2id.get(x_ij, oov_id)
  else:
    batch_w = None

  if char2id is not None:
    bow_id, eow_id, oov_id, pad_id = char2id.get('<eow>', None), char2id.get('<bow>', None), char2id.get(oov, None), char2id.get(pad, None)

    assert bow_id is not None and eow_id != None and oov_id is not None and pad_id is not None

    if config['token_embedder']['name'].lower() == 'cnn':
      max_chars = config['token_embedder']['max_characters_per_token']
      assert max([len(w) for i in lst for w in x[i]]) + 2 <= max_chars
    elif config['token_embedder']['name'].lower() == 'lstm':
      max_chars = max([len(w) for i in lst for w in x[i]]) + 2  # counting the <bow> and <eow>

    batch_c = torch.LongTensor(batch_size, max_len, max_chars).fill_(pad_id)

    for i, x_i in enumerate(x):
      for j, x_ij in enumerate(x_i):
        batch_c[i][j][0] = bow_id
        if x_ij == '<bos>' or x_ij == '<eos>':
          batch_c[i][j][1] = char2id.get(x_ij)
          batch_c[i][j][2] = eow_id
        else:
          for k, c in enumerate(x_ij):
            batch_c[i][j][k + 1] = char2id.get(c, oov_id)
          batch_c[i][j][len(x_ij) + 1] = eow_id
  else:
    batch_c = None

  masks = [torch.LongTensor(batch_size, max_len).fill_(0), [], []]

  for i, x_i in enumerate(x):
    for j in range(len(x_i)):
      masks[0][i][j] = 1
      if j + 1 < len(x_i):
        masks[1].append(i * max_len + j)
      if j > 0:
        masks[2].append(i * max_len + j)

  assert len(masks[1]) <= batch_size * max_len
  assert len(masks[2]) <= batch_size * max_len

  masks[1] = torch.LongTensor(masks[1])
  masks[2] = torch.LongTensor(masks[2])

  #print(batch_w.view(-1).size(0), masks[1].view(-1).size(0), masks[2].view(-1).size(0))
  return batch_w, batch_c, lens, masks


# shuffle training examples and create mini-batches
def create_batches(x, batch_size, word2id, char2id, config, perm=None, shuffle=True, sort=True, use_cuda=False, text=None):
  lst = perm or list(range(len(x)))
  if shuffle:
    random.shuffle(lst)

  if sort:
    lst.sort(key=lambda l: -len(x[l]))

  x = [x[i] for i in lst]
  if text is not None:
    text = [text[i] for i in lst]

  sum_len = 0.0
  batches_w, batches_c, batches_lens, batches_masks, batches_text = [], [], [], [], []
  size = batch_size
  nbatch = (len(x) - 1) // size + 1
  for i in range(nbatch):
    start_id, end_id = i * size, (i + 1) * size
    bw, bc, blens, bmasks = create_one_batch(x[start_id: end_id], word2id, char2id, config,
                                             sort=sort, use_cuda=use_cuda)
    sum_len += sum(blens)
    batches_w.append(bw)
    batches_c.append(bc)
    batches_lens.append(blens)
    batches_masks.append(bmasks)
    if text is not None:
      batches_text.append(text[start_id: end_id])

  if sort:
    perm = list(range(nbatch))
    random.shuffle(perm)
    batches_w = [batches_w[i] for i in perm]
    batches_c = [batches_c[i] for i in perm]
    batches_lens = [batches_lens[i] for i in perm]
    batches_masks = [batches_masks[i] for i in perm]
    if text is not None:
      batches_text = [batches_text[i] for i in perm]

  logging.info("{} batches, avg len: {:.1f}".format(nbatch, sum_len / len(x)))
  if text is not None:
    return batches_w, batches_c, batches_lens, batches_masks, batches_text
  return batches_w, batches_c, batches_lens, batches_masks


class Model(nn.Module):
  def __init__(self, config, word_emb_layer, char_emb_layer, use_cuda=False):
    super(Model, self).__init__()
    self.use_cuda = use_cuda
    self.config = config

    if config['token_embedder']['name'].lower() == 'cnn':
      self.token_embedder = ConvTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)
    elif config['token_embedder']['name'].lower() == 'lstm':
      self.token_embedder = LstmTokenEmbedder(config, word_emb_layer, char_emb_layer, use_cuda)

    if config['encoder']['name'].lower() == 'elmo':
      self.encoder = ElmobiLm(config, use_cuda)
    elif config['encoder']['name'].lower() == 'lstm':
      self.encoder = LstmbiLm(config, use_cuda)

    self.output_dim = config['encoder']['projection_dim']

  def forward(self, word_inp, chars_package, mask_package):
    token_embedding = self.token_embedder(word_inp, chars_package, (mask_package[0].size(0), mask_package[0].size(1)))
    if self.config['encoder']['name'] == 'elmo':
      mask = Variable(mask_package[0]).cuda() if self.use_cuda else Variable(mask_package[0])
      encoder_output = self.encoder(token_embedding, mask)
      sz = encoder_output.size()
      token_embedding = torch.cat([token_embedding, token_embedding], dim=2).view(1, sz[1], sz[2], sz[3])
      encoder_output = torch.cat([token_embedding, encoder_output], dim=0)
    elif self.config['encoder']['name'] == 'lstm':
      encoder_output = self.encoder(token_embedding)
    return encoder_output

  def load_model(self, path):
    self.token_embedder.load_state_dict(torch.load(os.path.join(path, 'token_embedder.pkl'), map_location=lambda storage, loc: storage))
    self.encoder.load_state_dict(torch.load(os.path.join(path, 'encoder.pkl'), map_location=lambda storage, loc: storage))


def test_main():
  cmd = argparse.ArgumentParser('The testing components of')
  cmd.add_argument('--gpu', default=-1, type=int, help='use id of gpu, -1 if cpu.')
  cmd.add_argument('--input_format', default='plain', choices=('plain', 'conll', 'conll_char', 'conll_char_vi'),
                   help='the input format.')
  cmd.add_argument("--input", help="the path to the raw text file.")
  cmd.add_argument('--output_ave', help='the path to the average embedding file.')
  cmd.add_argument('--output_lstm', help='the path to the 1st lstm-output embedding file.')

  cmd.add_argument("--model", required=True, help="path to save model")
  cmd.add_argument("--batch_size", "--batch", type=int, default=1, help='the batch size.')

  args = cmd.parse_args(sys.argv[2:])

  if args.gpu >= 0:
    torch.cuda.set_device(args.gpu)
  use_cuda = args.gpu >= 0 and torch.cuda.is_available()

  args2 = dict2namedtuple(json.load(codecs.open(os.path.join(args.model, 'config.json'), 'r', encoding='utf-8')))

  with open(args2.config_path, 'r') as fin:
    config = json.load(fin)

  if config['token_embedder']['char_dim'] > 0:
    char_lexicon = {}
    with codecs.open(os.path.join(args.model, 'char.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        char_lexicon[token] = int(i)
    char_emb_layer = EmbeddingLayer(config['token_embedder']['char_dim'], char_lexicon, fix_emb=False, embs=None)
    logging.info('char embedding size: ' + str(len(char_emb_layer.word2id)))
  else:
    char_lexicon = None
    char_emb_layer = None

  if config['token_embedder']['word_dim'] > 0:
    word_lexicon = {}
    with codecs.open(os.path.join(args.model, 'word.dic'), 'r', encoding='utf-8') as fpi:
      for line in fpi:
        tokens = line.strip().split('\t')
        if len(tokens) == 1:
          tokens.insert(0, '\u3000')
        token, i = tokens
        word_lexicon[token] = int(i)
    word_emb_layer = EmbeddingLayer(config['token_embedder']['word_dim'], word_lexicon, fix_emb=False, embs=None)
    logging.info('word embedding size: ' + str(len(word_emb_layer.word2id)))
  else:
    word_lexicon = None
    word_emb_layer = None

  model = Model(config, word_emb_layer, char_emb_layer, use_cuda)

  if use_cuda:
    model.cuda()

  logging.info(str(model))
  model.load_model(args.model)
  if config['token_embedder']['name'].lower() == 'cnn':
    if args.input_format == 'plain':
      test, text = read_corpus(args.input, config['token_embedder']['max_characters_per_token'])
    elif args.input_format == 'conll':
      test, text = read_conll_corpus(args.input, config['token_embedder']['max_characters_per_token'])
    elif args.input_format == 'conll_char':
      test, text = read_conll_char_corpus(args.input, config['token_embedder']['max_characters_per_token'])
    else:
      test, text = read_conll_char_vi_corpus(args.input, config['token_embedder']['max_characters_per_token'])
  elif config['token_embedder']['name'].lower() == 'lstm':
    if args.input_format == 'plain':
      test, text = read_corpus(args.input)
    elif args.input_format == 'conll':
      test, text = read_conll_corpus(args.input)
    elif args.input_format == 'conll_char':
      test, text = read_conll_char_corpus(args.input)
    else:
      test, text = read_conll_char_vi_corpus(args.input)

  test_w, test_c, test_lens, test_masks, test_text = create_batches(
    test, args.batch_size, word_lexicon, char_lexicon, config, use_cuda=use_cuda, text=text)

  print(max([len(x) for x in test]))

  model.eval()

  sent_set = set()

  cnt = 0

  fout_ave = h5py.File(args.output_ave, 'w') if args.output_ave is not None else None
  fout_lstm = h5py.File(args.output_lstm, 'w') if args.output_lstm is not None else None

  for w, c, lens, masks, texts in zip(test_w, test_c, test_lens, test_masks, test_text):
    output = model.forward(w, c, masks)
    for i, text in enumerate(texts):
      sent = '\t'.join(text)
      sent = sent.replace('.', '$period$')
      sent = sent.replace('/', '$backslash$')
      if sent in sent_set:
        continue
      sent_set.add(sent)
      if config['encoder']['name'].lower() == 'lstm':
        data = output[i, 1:lens[i]-1, :].data
        if use_cuda:
          data = data.cpu()
        data = data.numpy()
      elif config['encoder']['name'].lower() == 'elmo':
        data = output[:, i, 1:lens[i]-1, :].data
        if use_cuda:
          data = data.cpu()
        data = data.numpy()
      if fout_ave is not None:
        data_ave = np.average(data, axis=0)
        fout_ave.create_dataset(
          sent,
          data_ave.shape, dtype='float32',
          data=data_ave
        )
      if fout_lstm is not None:
        data_lstm = data[1]
        fout_lstm.create_dataset(
          sent,
          data_lstm.shape, dtype='float32',
          data=data_lstm
        )
      cnt += 1
      if cnt % 1000 == 0:
        logging.info('Finished {0} sentences.'.format(cnt))
  if fout_ave is not None:
    fout_ave.close()
  if fout_lstm is not None:
    fout_lstm.close()


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == 'test':
    test_main()
  else:
    print('Usage: {0} [test] [options]'.format(sys.argv[0]), file=sys.stderr)
