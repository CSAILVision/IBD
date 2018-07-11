import torch
import torch.nn as nn
from torchvision import models
import settings
import string
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
from loader.caption_helper import CaptionGenerator
import numpy as np

__UNK_TOKEN = 'UNK'
__PAD_TOKEN = 'PAD'
__EOS_TOKEN = 'EOS'

def simple_tokenize(captions):
    processed = []
    for j, s in enumerate(captions):
        txt = str(s).lower().translate(string.punctuation).strip().split()
        processed.append(txt)
    return processed

def create_target(vocab):
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    unk = word2idx[__UNK_TOKEN]

    def get_caption(captions):
        captions = simple_tokenize(captions)
        caption = captions[0]
        targets = []
        for w in caption:
            targets.append(word2idx.get(w, unk))
        return torch.Tensor(targets)
    return get_caption


def create_batches(vocab, max_length=settings.BATCH_SIZE):
    padding = vocab.index(__PAD_TOKEN)
    eos = vocab.index(__EOS_TOKEN)

    def collate(img_cap):
        imgs, caps = img_cap
        imgs = torch.cat([img.unsqueeze(0) for img in imgs], 0)
        lengths = [min(len(c) + 1, max_length) for c in caps]
        batch_length = max(lengths)
        cap_tensor = torch.LongTensor(batch_length, len(caps)).fill_(padding)
        for i, c in enumerate(caps):
            end_cap = lengths[i]
            if end_cap < batch_length:
                cap_tensor[end_cap, i] = eos

            cap_tensor[:end_cap, i].copy_(c[:end_cap])

        return (imgs, (cap_tensor, lengths))
    return collate



class CaptionModel(nn.Module):

    def __init__(self, cnn=None, vocab=None, voc_size=10003, embedding_size=256, rnn_size=256, num_layers=2,
                 share_embedding_weights=False):
        super(CaptionModel, self).__init__()
        self.vocab = vocab
        if cnn:
            self.cnn = cnn
        else:
            self.cnn = models.__dict__[settings.CNN_MODEL](pretrained=True)
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embedding_size)
        self.rnn = nn.LSTM(embedding_size, rnn_size, num_layers=num_layers)
        self.classifier = nn.Linear(rnn_size, voc_size)
        self.embedder = nn.Embedding(voc_size, embedding_size)
        if share_embedding_weights:
            self.embedder.weight = self.classifier.weight


    def forward(self, imgs, captions ):
        captions = torch.LongTensor(captions).transpose(1,0)[:,:-1]
        if settings.GPU:
            captions = captions.cuda()
        embeddings = self.embedder(Variable(captions))

        img_feats = self.cnn(imgs).unsqueeze(1)
        embeddings = torch.cat([img_feats, embeddings], 1)
        feats, state = self.rnn(embeddings.transpose(1,0))
        pred = self.classifier(feats.transpose(1,0)).max(2)[0]

        return pred, state

    def generate(self, img,eos_token='EOS',max_caption_length=20):
        cap_gen = CaptionGenerator(embedder=self.embedder,
                                   rnn=self.rnn,
                                   classifier=self.classifier,
                                   eos_id=self.vocab.index(eos_token),
                                   max_caption_length=max_caption_length)
        img_feats = self.cnn(img).unsqueeze(0)
        sentences, score, vars = cap_gen.beam_search(img_feats)
        words = [[self.vocab[sentences[0][wid][b]] for wid in range(max_caption_length)] for b in
               range(img.data.shape[0])]

        return sentences, words

    def save_checkpoint(self, filename):
        torch.save({'embedder_dict': self.embedder.state_dict(),
                    'rnn_dict': self.rnn.state_dict(),
                    'cnn_dict': self.cnn.state_dict(),
                    'classifier_dict': self.classifier.state_dict(),
                    'vocab': self.vocab},
                   filename)

    def load_checkpoint(self, filename):
        if not settings.GPU:
            cpnt = torch.load(filename, map_location=lambda storage, loc: storage )
        else:
            cpnt = torch.load(filename)

        if 'cnn_dict' in cpnt:
            self.cnn.load_state_dict(cpnt['cnn_dict'])
        self.embedder.load_state_dict(cpnt['embedder_dict'])
        self.rnn.load_state_dict(cpnt['rnn_dict'])
        self.vocab=cpnt['vocab']
        self.classifier.load_state_dict(cpnt['classifier_dict'])

    def finetune_cnn(self, allow=True):
        for p in self.cnn.parameters():
            p.requires_grad = allow
        for p in self.cnn.fc.parameters():
            p.requires_grad = True
