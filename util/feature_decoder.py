import os
import time

import numpy as np
import torch
from torch import nn as nn, optim as optim
from torch.autograd import Variable as V
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
import settings
from loader.feature_loader import feature_loader, concept_loader, concept_loader_factory, ConceptDataset


class FeatureClassifier(nn.Module):
    def __init__(self,):
        super(FeatureClassifier, self).__init__()
        self.epoch = 0

    def forward(self, input):
        raise NotImplementedError

    def load_snapshot(self, epoch, unbiased=True):
        self.epoch = epoch
        self.load_state_dict(torch.load(os.path.join(settings.OUTPUT_FOLDER, "snapshot", "%d.pth" % epoch)))
        if unbiased:
            w = self.fc.weight.data
            w_ub = w - w.mean(1)[:, None]
            self.fc.weight.data.copy_(w_ub)

    def save_snapshot(self, epoch):
        torch.save(self.state_dict(), os.path.join(settings.OUTPUT_FOLDER, "snapshot", "%d.pth" % epoch))

    def val(self):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

class SemanticFeatureClassifier(FeatureClassifier):
    def __init__(self, feat_len, concept_size):
        super(FeatureClassifier, self).__init__()
        self.fc = nn.Linear(feat_len, concept_size)

    def forward(self, input):
        return self.fc(input)

    def val(self, feature, layer, fo):
        filename = os.path.join(settings.OUTPUT_FOLDER, "%s-val-accuracy.npy" % layer)
        if os.path.exists(filename):
            label_count, label_top1_accuracy, label_top5_accuracy, epoch_error, top1, top5 = np.load(filename)
        else:
            loss = nn.CrossEntropyLoss()
            if settings.GPU:
                loss.cuda()
            feat_loader_train, feat_loader_test = feature_loader(feature, layer, fo.data, len(fo.data.label), split=True)
            epoch_error = 0
            label_count = torch.zeros(len(fo.data.label))
            label_top1_accuracy = torch.zeros(len(fo.data.label)).fill_(-1e-10)
            label_top5_accuracy = torch.zeros(len(fo.data.label)).fill_(-1e-10)
            start_time = time.time()
            last_batch_time = start_time
            for i, (feat, label) in enumerate(feat_loader_test):
                if type(feat) == int:
                    continue
                batch_time = time.time()
                rate = i * settings.FEAT_BATCH_SIZE / (batch_time - start_time + 1e-15)
                batch_rate = settings.FEAT_BATCH_SIZE / (batch_time - last_batch_time + 1e-15)
                last_batch_time = batch_time

                feat_var = V(feat, requires_grad=True)
                if settings.GPU:
                    feat_var.cuda()
                out = self.forward(feat_var)
                err = loss(out, V(label))
                epoch_error += 1 / (i + 1) * (err.data[0] - epoch_error)
                top5 = label[:, None] == torch.topk(out, 5, 1)[1].data
                for b_i in range(len(label)):
                    label_count[label[b_i]] += 1
                    label_top1_accuracy[label[b_i]] += 1 / label_count[label[b_i]] * (top5[b_i, 0] - label_top1_accuracy[label[b_i]])
                    label_top5_accuracy[label[b_i]] += 1 / label_count[label[b_i]] * (float(top5[b_i].sum()) - label_top5_accuracy[label[b_i]])
                top1 = ((label_top1_accuracy >= 0).float() * label_top1_accuracy).sum() / (label_top1_accuracy >= 0).sum()
                top5 = ((label_top5_accuracy >= 0).float() * label_top5_accuracy).sum() / (label_top5_accuracy >= 0).sum()
                print("val epoch [%d/%d]: batch error %.4f, overall error %.4f, top <1> %.4f, top <5> %.4f, item per second %.4f, %.4f" % (
                    i + 1, len(feat_loader_test), err.data[0], epoch_error, top1, top5, batch_rate, rate))
            if settings.GPU:
                label_count, label_top1_accuracy, label_top5_accuracy = label_count.cpu().numpy(), label_top1_accuracy.cpu().numpy(), label_top5_accuracy.cpu().numpy()
            else:
                label_count, label_top1_accuracy, label_top5_accuracy = label_count.numpy(), label_top1_accuracy.numpy(), label_top5_accuracy.numpy()
            np.save(filename, (label_count, label_top1_accuracy, label_top5_accuracy, epoch_error, top1, top5))
        import matplotlib.pyplot as plt
        plt.figure(figsize=(24,4))
        plt.plot(range(len(label_top1_accuracy[label_top1_accuracy >= 0])), label_top1_accuracy[label_top1_accuracy >= 0], label='top 1')
        plt.plot(range(len(label_top5_accuracy[label_top5_accuracy >= 0])), label_top5_accuracy[label_top5_accuracy >= 0], label='top 5')
        plt.legend(loc='upper right')
        plt.title("%s \n error:%.4f, top <1> %.4f, top <5> %.4f" % (settings.CNN_MODEL, epoch_error, top1, top5))
        plt.tight_layout()
        plt.savefig(os.path.join(settings.OUTPUT_FOLDER, 'html', 'image', "accuracy_distribute.jpg"))

    def train(self, feature, layer, fo):
        optimizer = optim.SGD(self.parameters(), lr=0.02)
        # loss = nn.MSELoss()
        loss = nn.CrossEntropyLoss()
        if settings.GPU:
            loss.cuda()
        feat_loader_train, feat_loader_test = concept_loader(feature, layer, fo.data, len(fo.data.label))
        for epoch in range(self.epoch+1, settings.EPOCHS+1):

            # training
            training_epoch_error = 0
            start_time = time.time()
            last_batch_time = start_time
            for i, (feat, label) in enumerate(feat_loader_train):
                if type(feat) == int:
                    continue
                # feat = feat.view((settings.FEAT_BATCH_SIZE * settings.SEG_RESOLUTION ** 2, -1))
                # label = label.view((settings.FEAT_BATCH_SIZE * settings.SEG_RESOLUTION ** 2)).long()
                batch_time = time.time()
                rate = i * settings.FEAT_BATCH_SIZE / (batch_time - start_time + 1e-15)
                batch_rate = settings.FEAT_BATCH_SIZE / (batch_time - last_batch_time + 1e-15)
                last_batch_time = batch_time

                feat_var = V(feat, requires_grad=True)
                if settings.GPU:
                    feat_var.cuda()
                out = self.forward(feat_var)
                err = loss(out, V(label))
                training_epoch_error += 1 / (i + 1) * (err.data[0] - training_epoch_error)
                print("training epoch [%d/%d][%d/%d]: batch error %.6f, overall error %.6f, item per second %.4f, %.4f" % (
                epoch, settings.EPOCHS, i + 1, len(feat_loader_train), err.data[0], training_epoch_error, batch_rate, rate))
                optimizer.zero_grad()
                err.backward()
                optimizer.step()

            # validation
            if feat_loader_test:
                val_epoch_error = 0
                for i, (feat, label) in enumerate(feat_loader_test):
                    feat_var = V(feat, volatile=True)
                    if settings.GPU:
                        feat_var.cuda()
                    out = self.forward(feat_var)
                    err = loss(out, V(label))
                    val_epoch_error += 1 / (i + 1) * (err.data[0] - val_epoch_error)
                    print("validation epoch [%d/%d][%d/%d]: batch error %.6f, overall error %.6f" % (
                    epoch, settings.EPOCHS, i + 1, len(feat_loader_test), err.data[0], val_epoch_error))

            if epoch % settings.SNAPSHOT_FREQ == 0:
                self.save_snapshot(epoch)

class IndexLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True):
        super(IndexLinear, self).__init__(in_features, out_features, bias)

    def forward(self, input, id=None):
        if id is not None:
            return F.linear(input, self.weight[id:id+1, :], self.bias[id:id+1])
        else:
            return F.linear(input, self.weight, self.bias)

from torch.nn.modules.loss import _Loss
class NegWLoss(_Loss):

    def __init__(self, size_average=True, alpha=0.01):
        super(NegWLoss, self).__init__(size_average)
        self.alpha = alpha

    def forward(self, weight):
        return self.alpha * weight.mean()#F.relu(-weight).sum()



class SingleSigmoidFeatureClassifier(FeatureClassifier):
    def __init__(self, feature=None, layer=None, fo=None):
        super(SingleSigmoidFeatureClassifier, self).__init__()

        self.dataset = ConceptDataset(feature, layer, fo.data, len(fo.data.label), )
        # self.feat_loader = feature_loader(feature, layer, fo.data, len(fo.data.label))
        self.loader_factory = concept_loader_factory(feature, layer, fo.data, len(fo.data.label), concept_dataset=self.dataset)
        self.valid_concepts = self.dataset.concept_count.nonzero()[0]
        self.feat = feature
        self.layer_name = layer
        self.fo = fo
        self.concept_size = len(self.valid_concepts)
        self.display_epoch = 100
        if feature is None:
            self.fc = IndexLinear(1024, 660)
        else:
            self.fc = IndexLinear(feature.shape[1], self.concept_size)
        # self.fc.weight
        self.sig = nn.Sigmoid()

        self.loss_mse = nn.MSELoss()
        self.loss_weight = NegWLoss()
        self.optimizer = optim.SGD(self.parameters(), lr=1e-2)
        if settings.GPU:
            self.loss_mse.cuda()
            self.loss_weight.cuda()

    def forward(self, input, id=None):
        return self.sig(self.fc(input, id))

    def run(self):
        history = np.memmap(os.path.join(settings.OUTPUT_FOLDER, "mAP_table.mmap"), dtype=float, mode='w+', shape=(self.concept_size, settings.EPOCHS))
        neg_test_loaders = self.loader_factory.negative_test_concept_loader(sample_ratio=20, verbose=False)
        neg_scores = None
        for epoch in range(self.epoch, settings.EPOCHS):
            concept_train_loaders, concept_val_loaders = self.loader_factory.negative_mining_loader(neg_scores=neg_scores)
            neg_scores = [None] * self.concept_size
            for c_i in range(self.concept_size):
                train_loader = concept_train_loaders[c_i]
                val_loader = concept_val_loaders[c_i]
                test_loader = neg_test_loaders[c_i]
                self.train(train_loader, c_i, epoch)
                neg_scores[c_i] = self.test(test_loader, c_i, epoch)
                history[c_i, epoch] = self.val(val_loader, c_i, epoch)
            self.save_snapshot(epoch)
            np.save(os.path.join(settings.OUTPUT_FOLDER, "snapshot", "neg_scores.npy"), neg_scores)

    def run_naive(self):
        # history = np.memmap(os.path.join(settings.OUTPUT_FOLDER, "mAP_table.mmap"), dtype=float, mode='w+', shape=(self.concept_size, settings.EPOCHS))
        for epoch in range(self.epoch, settings.EPOCHS):
            concept_train_loaders, concept_val_loaders = self.loader_factory.random_concept_loader()
            for c_i in range(self.concept_size):
                train_loader = concept_train_loaders[c_i]
                # val_loader = concept_val_loaders[c_i]
                self.train(train_loader, c_i, epoch)
                # history[c_i, epoch] = self.val(val_loader, c_i, epoch)
            self.save_snapshot(epoch)

    def run_fix_eval(self):
        concept_val_loaders = self.loader_factory.fixed_val_loader()
        aps = np.zeros(self.concept_size)
        for c_i in range(self.concept_size):
            val_loader = concept_val_loaders[c_i]
            aps[c_i] = self.val(val_loader, c_i, settings.EPOCHS)
        np.save(os.path.join(settings.OUTPUT_FOLDER, "mAP_val.npy"), aps)
        print("mAP {:4.4f}%".format(aps.mean()))

    def val(self, val_loader, c_i, epoch):
        val_epoch_error = 0
        val_label = np.zeros(len(val_loader.dataset))
        val_score = np.zeros(len(val_loader.dataset))
        for i, (ind, feat, label) in enumerate(val_loader):
            start_ind = i * settings.BATCH_SIZE
            end_ind = i * settings.BATCH_SIZE + len(feat)
            feat_var = V(feat, requires_grad=True)
            if settings.GPU:
                feat_var.cuda()
            out = self.forward(feat_var, c_i)
            err = self.loss_mse(out, V(label))
            val_epoch_error += 1 / (i + 1) * (err.data[0] - val_epoch_error)
            val_label[start_ind:end_ind] = label.numpy()
            val_score[start_ind:end_ind] = out[:,0].data.numpy()
            if i % self.display_epoch == 0:
                print("val epoch [%d/%d][%d/%d][%d/%d]: batch error %.6f, overall error %.6f" % (
                    c_i + 1, len(self.valid_concepts),
                    epoch, settings.EPOCHS, i + 1, len(val_loader), err.data[0], val_epoch_error,
                ))
        AP = average_precision_score(val_label, val_score)
        print("Concept {:d}, epoch{:d}, AP: {:4.2f}%".format(c_i, epoch, AP * 100))
        return AP

    # def run_val_resize(self, size):
    #

    def train(self, train_loader, c_i, epoch):
        training_epoch_error = 0
        start_time = time.time()
        last_batch_time = start_time
        for i, (ind, feat, label) in enumerate(train_loader):
            batch_time = time.time()
            rate = i * settings.FEAT_BATCH_SIZE / (batch_time - start_time + 1e-15)
            batch_rate = settings.FEAT_BATCH_SIZE / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time

            feat_var = V(feat, requires_grad=True)
            if settings.GPU:
                feat_var.cuda()
            out = self.forward(feat_var, c_i)
            err_mse = self.loss_mse(out, V(label))
            # err_weight = self.loss_weight(self.fc.weight[c_i])
            err_weight = V(torch.FloatTensor([0]))
            err = err_mse #+ err_weight
            training_epoch_error += 1 / (i + 1) * (err_mse.data[0] - training_epoch_error)
            if i % self.display_epoch == 0:
                print("training epoch [%d/%d][%d/%d][%d/%d]: mse error %.5f, weight loss %.5f overall mse error %.5f, item per second %.4f, %.4f" % (
                        c_i + 1, len(self.valid_concepts),
                        epoch, settings.EPOCHS, i + 1, len(train_loader), err_mse.data[0], err_weight.data[0], training_epoch_error,
                        batch_rate, rate))
            self.optimizer.zero_grad()
            err.backward()
            self.optimizer.step()
            # self.fc.weight[c_i].data.copy_(F.relu(self.fc.weight[c_i]).data)

    def test(self, test_loader, c_i, epoch):
        training_epoch_error = 0
        start_time = time.time()
        last_batch_time = start_time
        prediction_scores = np.zeros(len(test_loader.dataset))
        for i, (ind, feat, label) in enumerate(test_loader):
            batch_time = time.time()
            rate = i * settings.FEAT_BATCH_SIZE / (batch_time - start_time + 1e-15)
            batch_rate = settings.FEAT_BATCH_SIZE / (batch_time - last_batch_time + 1e-15)
            last_batch_time = batch_time

            start_ind = i * settings.BATCH_SIZE
            end_ind = i * settings.BATCH_SIZE + len(feat)

            feat_var = V(feat, requires_grad=True)
            if settings.GPU:
                feat_var.cuda()
            out = self.forward(feat_var, c_i)
            err = self.loss_mse(out, V(label))
            prediction_scores[start_ind:end_ind] = out.data.squeeze().numpy()
            training_epoch_error += 1 / (i + 1) * (err.data[0] - training_epoch_error)
            if i % self.display_epoch == 0:
                print(
                    "testing epoch [%d/%d][%d/%d][%d/%d]: batch error %.6f, overall error %.6f, item per second %.4f, %.4f" % (
                        c_i + 1, len(self.valid_concepts),
                        epoch, settings.EPOCHS, i + 1, len(test_loader), err.data[0], training_epoch_error,
                        batch_rate, rate))
        return prediction_scores
