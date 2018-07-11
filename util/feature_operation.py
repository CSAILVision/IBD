
import os
import torch
from torch.autograd import Variable as V
import torch.nn as nn
from scipy.misc import imresize, imread, imsave
import cv2
from PIL import Image
import settings
import numpy as np
import pickle
import torch.nn.functional as F
from sklearn.manifold import TSNE, SpectralEmbedding
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
from loader.data_loader import SegmentationData, SegmentationPrefetcher
from visualize.plot import random_color
from util.places365_categories import places365_categories
from util.imagenet_categories import imagenet_categories
from util.image_operation import *



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

grad_blobs = []
def hook_grad(module, grad_input, grad_output):
    grad_blobs.append(grad_output[0].data.cpu().numpy())

class CAMWrapper(nn.Module):
    def __init__(self, model):
        super(CAMWrapper,self).__init__()
        assert type(list(model.children())[-1]).__name__ == 'Linear'
        self.layers = list(model.children())
        if type(list(model.children())[-2]).__name__ == 'AvgPool2d':
            self.features = nn.Sequential(*self.layers[:-2])
        else:
            self.features = nn.Sequential(*self.layers[:-1])
        self.conv1x1 = nn.Conv2d(self.layers[-1].in_features, self.layers[-1].out_features, 1, bias=False)
        self.conv1x1.weight = nn.Parameter(self.layers[-1].weight.data.unsqueeze(2).unsqueeze(3))

    def forward(self, input):
        feature_map = self.features(input)
        cam_map = self.conv1x1(feature_map)
        return cam_map


class FeatureOperator:

    def __init__(self):
        if not os.path.exists(settings.OUTPUT_FOLDER):
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'html', 'image'))
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'snapshot'))
            os.makedirs(os.path.join(settings.OUTPUT_FOLDER, 'sample_cache'))
        self.data = SegmentationData(settings.DATA_DIRECTORY, categories=settings.CATAGORIES)
        self.loader = SegmentationPrefetcher(self.data,categories=['image'],once=True,batch_size=settings.BATCH_SIZE)
        self.mean = [109.5388,118.6897,124.6901]


    def val(self, model):
        # val_loader = places365_imagenet_loader('val')
        import torch.nn as nn
        criterion = nn.CrossEntropyLoss().cuda()

        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        for i, (input, target) in enumerate(val_loader):

            target = target.cuda()
            input = input.cuda()
            input_var = torch.autograd.Variable(input, volatile=True)
            target_var = torch.autograd.Variable(target, volatile=True)

            # compute output
            fc_output = model(input_var)

            loss = criterion(fc_output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(fc_output.data, target, topk=(1, 5))
            if torch.__version__.startswith('0.4'):
                losses.update(loss.item(), input.size(0))
            else:
                losses.update(loss.data[0], input.size(0))

            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            if i % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                    i, len(val_loader), loss=losses,
                    top1=top1, top5=top5))

        val_acc = top1.avg

        print('VAL Prec@1 %.3f ' % (val_acc))

    def feature_extraction(self, model=None, memmap=True):
        loader = self.loader
        # extract the max value activaiton for each image
        maxfeatures = [None] * len(settings.FEATURE_NAMES)
        wholefeatures = [None] * len(settings.FEATURE_NAMES)
        features_size = [None] * len(settings.FEATURE_NAMES)
        features_size_file = os.path.join(settings.OUTPUT_FOLDER, "feature_size.npy")

        if memmap:
            skip = True
            mmap_files = [os.path.join(settings.OUTPUT_FOLDER, "%s.mmap" % feature_name)  for feature_name in  settings.FEATURE_NAMES]
            mmap_max_files = [os.path.join(settings.OUTPUT_FOLDER, "%s_max.mmap" % feature_name) for feature_name in settings.FEATURE_NAMES]
            if os.path.exists(features_size_file):
                features_size = np.load(features_size_file)
            else:
                skip = False
            for i, (mmap_file, mmap_max_file) in enumerate(zip(mmap_files,mmap_max_files)):
                if os.path.exists(mmap_file) and os.path.exists(mmap_max_file) and features_size[i] is not None:
                    print('loading features %s' % settings.FEATURE_NAMES[i])
                    wholefeatures[i] = np.memmap(mmap_file, dtype=np.float32,mode='r', shape=tuple(features_size[i]))
                    maxfeatures[i] = np.memmap(mmap_max_file, dtype=np.float32, mode='r', shape=tuple(features_size[i][:2]))
                else:
                    print('file missing, loading from scratch')
                    skip = False
            if skip:
                return wholefeatures, maxfeatures

        num_batches = (len(loader.indexes) + loader.batch_size - 1) / loader.batch_size
        for batch_idx,batch in enumerate(loader.tensor_batches(bgr_mean=self.mean)):
            del features_blobs[:]
            print('extracting feature from batch %d / %d' % (batch_idx+1, num_batches))

            input = batch[0]
            if settings.CAFFE_MODEL:
                input = torch.from_numpy(input.copy())
            else:
                input = torch.from_numpy(input[:, ::-1, :, :].copy())
                input.div_(255.0 * 0.224)
            batch_size = len(input)


            if settings.GPU:
                input = input.cuda()
            input_var = V(input,volatile=True)
            if settings.APP == "classification":
                output = model.forward(input_var)
            else:
                output = model.cnn.forward(input_var)
            while np.isnan(output.data.max()):
                print("nan") #which I have no idea why it will happen
                del features_blobs[:]
                output = model.forward(input_var)
            if maxfeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (len(loader.indexes), feat_batch.shape[1])
                    if memmap:
                        maxfeatures[i] = np.memmap(mmap_max_files[i],dtype=np.float32,mode='w+',shape=size_features)
                    else:
                        maxfeatures[i] = np.zeros(size_features)
            if len(feat_batch.shape) == 4 and wholefeatures[0] is None:
                # initialize the feature variable
                for i, feat_batch in enumerate(features_blobs):
                    size_features = (
                    len(loader.indexes), feat_batch.shape[1], feat_batch.shape[2], feat_batch.shape[3])
                    features_size[i] = size_features
                    if memmap:
                        wholefeatures[i] = np.memmap(mmap_files[i], dtype=np.float32, mode='w+', shape=size_features)
                    else:
                        wholefeatures[i] = np.zeros(size_features)
            np.save(features_size_file, features_size)
            start_idx = batch_idx*settings.BATCH_SIZE
            end_idx = min((batch_idx+1)*settings.BATCH_SIZE, len(loader.indexes))
            for i, feat_batch in enumerate(features_blobs):
                if len(feat_batch.shape) == 4:
                    wholefeatures[i][start_idx:end_idx] = feat_batch
                    maxfeatures[i][start_idx:end_idx] = np.max(np.max(feat_batch,3),2)
                elif len(feat_batch.shape) == 3:
                    maxfeatures[i][start_idx:end_idx] = np.max(feat_batch, 2)
                elif len(feat_batch.shape) == 2:
                    maxfeatures[i][start_idx:end_idx] = feat_batch
        if len(feat_batch.shape) == 2:
            wholefeatures = maxfeatures
        return wholefeatures,maxfeatures

    def vqa_feature_extraction(self, model, org_img, q, q_len, a, ):
        del features_blobs[:]
        img = np.array(org_img, dtype=np.float32)
        if (img.ndim == 2):
            img = np.repeat(img[:, :, None], 3, axis=2)
        img -= np.array(self.mean)[::-1]
        img = img.transpose((2, 0, 1))
        if settings.CAFFE_MODEL:
            input = torch.from_numpy(img[None, ::-1, :, :].copy())
        else:
            input = torch.from_numpy(img[None, :, :, :].copy())
            input.div_(255.0 * 0.224)
        input_var = V(input, requires_grad=True)
        if settings.GPU:
            input_var = input_var.cuda()
        model.cnn.forward(input_var)
        img_feat = features_blobs[0]
        v = V(torch.from_numpy(img_feat).cuda(async=True), requires_grad=True)
        q = V(q.cuda(async=True), requires_grad=False)
        # a = V(a.cuda(async=True), requires_grad=False)
        q_len = V(q_len.cuda(async=True), requires_grad=False)
        out = model(v, q, q_len)
        val, ind = out.max(1)
        val.backward(torch.FloatTensor([1]).cuda())
        img_grad = v.grad.data.cpu().numpy()
        return img_feat[0].transpose(1, 2, 0), img_grad[0].transpose(1, 2, 0), ind, None

    def imagecap_feature_extraction(self, model, org_img):
        del features_blobs[:]
        img = np.array(org_img, dtype=np.float32)
        if (img.ndim == 2):
            img = np.repeat(img[:, :, None], 3, axis=2)
        img -= np.array(self.mean)[::-1]
        img = img.transpose((2, 0, 1))
        if settings.CAFFE_MODEL:
            input = torch.from_numpy(img[None, ::-1, :, :].copy())
        else:
            input = torch.from_numpy(img[None, :, :, :].copy())
            input.div_(255.0 * 0.224)
        input_var = V(input, requires_grad=True)
        if settings.GPU:
            input_var = input_var.cuda()
        model.cnn.forward(input_var)
        img_feat = features_blobs[0]
        v = V(torch.from_numpy(img_feat).cuda(async=True), requires_grad=True)
        sents, caps = model.generate(input_var)
        out, _ = model(input_var, sents[0])
        sents = np.array(sents).ravel()
        if np.where(sents == 10002)[0].__len__() != 0:
            sents = sents[:np.where(sents == 10002)[0][0]]

        # words_ind = sents.argsort()[:-3:-1]
        w = sents.argmax()
        onehot = torch.zeros(out[0].size())
        onehot[w] = 1
        del grad_blobs[:]
        model.cnn.zero_grad()
        if settings.GPU:
            onehot = onehot.cuda()
        out[0].backward(onehot)
        img_feat = features_blobs[0][0].transpose(1, 2, 0)
        img_grad = grad_blobs[0][0].transpose(1, 2, 0)

        return img_feat, img_grad, w, sents

    def single_feature_extraction(self, model, org_img):
        del features_blobs[:]
        img = np.array(org_img, dtype=np.float32)
        if (img.ndim == 2):
            img = np.repeat(img[:, :, None], 3, axis=2)
        img -= np.array(self.mean)[::-1]
        img = img.transpose((2, 0, 1))
        if settings.CAFFE_MODEL:
            input = torch.from_numpy(img[None, ::-1, :, :].copy())
        else:
            input = torch.from_numpy(img[None, :, :, :].copy())
            input.div_(255.0 * 0.224)
        input_var = V(input, requires_grad=True)
        if settings.GPU:
            input_var = input_var.cuda()
        out = model.forward(input_var)
        onehot = torch.zeros(out.size())
        if settings.GPU:
            onehot = onehot.cuda()
        if torch.__version__.startswith('0.4'):
            onehot[0][out.max(1)[1].item()] = 1.0
        else:
            onehot[0][out.max(1)[1].data[0]] = 1.0

        if settings.DATASET == 'imagenet':
            if torch.__version__.startswith('0.4'):
                prediction = imagenet_categories[out.max(1)[1].item()]
            else:
                prediction = imagenet_categories[out.max(1)[1].data[0]]
        elif settings.DATASET == 'places365':
            if torch.__version__.startswith('0.4'):
                prediction = places365_categories[out.max(1)[1].item()]
            else:
                prediction = places365_categories[out.max(1)[1].data[0]]

        del grad_blobs[:]
        model.zero_grad()
        out.backward(onehot)
        img_feat = features_blobs[0][0].transpose(1, 2, 0)
        img_grad = grad_blobs[0][0].transpose(1, 2, 0)
        if torch.__version__.startswith('0.4'):
            return img_feat, img_grad, out.max(1)[1].item(), prediction
        else:
            return img_feat, img_grad, out.max(1)[1].data[0], prediction


    def weight_extraction(self, model, feat_clf):
        params = list(model.parameters())
        if settings.GPU:
            weight_softmax = params[-2].data.cpu().numpy()
            weight_clf = feat_clf.fc.weight.data.cpu().numpy()
        else:
            weight_softmax = params[-2].data.numpy()
            weight_clf = feat_clf.fc.weight.data.numpy()
        # weight_label = np.maximum(weight_softmax, 0)
        # weight_concept = np.maximum(weight_clf, 0)
        weight_label = weight_softmax
        weight_concept = weight_clf
        weight_label = weight_label / np.linalg.norm(weight_label, axis=1)[:, None]
        weight_concept = weight_concept / np.linalg.norm(weight_concept, axis=1)[:, None]
        return weight_label, weight_concept

    def weight_decompose(self, model, feat_clf, feat_labels=None):
        weight_label, weight_concept = self.weight_extraction(model, feat_clf)
        filename = os.path.join(settings.OUTPUT_FOLDER, "decompose.npy")
        if os.path.exists(filename):
            rankings, errvar, coefficients, residuals_T = np.load(filename)
        else:
            rankings, errvar, coefficients, residuals = self.decompose_Gram_Schmidt(weight_concept, weight_label, prediction_ind=None, MAX=settings.BASIS_NUM)
            np.save(filename, (rankings, errvar, coefficients, residuals.T))
            # for i in range(len(weight_label)):
            #     residuals[i] = weight_label[i] - np.matmul(ws[i][None, :], weight_concept[rankings[i, :].astype(int)])

        if settings.COMPRESSED_INDEX:
            try:
                feat_labels = [feat_labels[concept] for concept in feat_clf.valid_concepts]
            except Exception:
                feat_labels = [feat_labels[concept] for concept in np.load('cache/valid_concept.npy')]
        if settings.DATASET == "places365":
            model_labels = places365_categories
        elif settings.DATASET == "imagenet":
            model_labels = imagenet_categories
        for pi in range(len(rankings)):
            prediction = model_labels[pi]
            print(prediction, end=":\t")
            concept_inds = rankings[pi, :]
            for ci, concept_ind in enumerate(concept_inds):
                print("%s(%.2f) -> (%.2f)" % (feat_labels[concept_ind], coefficients[pi, ci], errvar[pi, ci]), end=",\t")
            print()

    def decompose_Gram_Schmidt(self, weight_concept, weight_label, prediction_ind=None, MAX=20):

        if prediction_ind is not None:
            if type(prediction_ind) == int:
                weight_label = weight_label[prediction_ind:prediction_ind+1, :]
            else:
                weight_label = weight_label[prediction_ind, :]

        rankings = np.zeros((len(weight_label), MAX), dtype=np.int32)
        errvar = np.zeros((len(weight_label), MAX))
        coefficients = np.zeros((len(weight_label), MAX + 2))
        residuals = np.zeros((len(weight_label), weight_concept.shape[1]))
        for label_id in range(len(weight_label)):
            if len(weight_label) > 10:
                print("decomposing label %d" % label_id)
            qo = weight_label[label_id].copy()
            residual = weight_label[label_id].copy()
            ortho_concepts = [(i, qc) for i, qc in enumerate(weight_concept)]
            basis = np.zeros((weight_label.shape[1], MAX + 2))
            for epoch in range(MAX):
                if MAX > 50:
                    print("epoch (%d/%d)" % (epoch, MAX))
                _, best_uc, best_index = max([(sum(uc * qo), uc, index) for index, uc in ortho_concepts])
                residual -= best_uc * sum(best_uc * residual)
                basis[:, epoch] = weight_concept[best_index]
                rankings[label_id][epoch] = best_index
                errvar[label_id][epoch] = np.linalg.norm(residual) ** 2#cosine_similarity(weight_label[label_id][None,:], (weight_label[label_id] - residual)[None,:])[:,0]#
                ortho_concepts = [(i, (uc - best_uc * sum(best_uc * uc)) / np.linalg.norm((uc - best_uc * sum(best_uc * uc))))
                                  for i, uc in ortho_concepts if i != best_index]
            positive_residual = np.maximum(residual, 0)
            negative_residual = -np.minimum(residual, 0)
            basis[:, MAX] = positive_residual / np.linalg.norm(positive_residual)
            basis[:, MAX + 1] = negative_residual / np.linalg.norm(negative_residual)
            residuals[label_id] = residual
            coefficients[label_id] = np.dot(np.linalg.pinv(basis), qo)
        return rankings, errvar, coefficients, residuals

    def decompose_cosine_similarity(self, weight_concept, weight_label, prediction_ind=None, MAX=7):
        if prediction_ind is not None:
            if type(prediction_ind) == int:
                weight_label = weight_label[prediction_ind:prediction_ind+1, :]
            else:
                weight_label = weight_label[prediction_ind, :]
        X = cosine_similarity(weight_label, weight_concept)
        rankings = X.argsort(1)[:, :-MAX-1:-1]
        scores = np.zeros((len(weight_label), MAX))
        ws = np.zeros((len(weight_label), MAX))
        for label_id in range(len(weight_label)):
            print("decomposing label %d" % label_id)
            for epoch in range(MAX):
                B = weight_label[label_id][None,:]
                A = weight_concept[rankings[label_id, :epoch + 1]]
                scores[label_id][epoch] = cosine_similarity(B, np.matmul(np.matmul(B, np.linalg.pinv(A)), A))
            ws[label_id] = np.matmul(B, np.linalg.pinv(A)).ravel()
        return rankings, scores, ws

    def ranking_gradient(self, weight_concept, img_grad,):
        # img_feat_resized_v = V(torch.FloatTensor(img_feat))
        # concept_predicted = feat_clf.fc(img_feat_resized_v)
        # concept_grad = feat_clf.fc.weight[None, :, :] * ((F.sigmoid(concept_predicted)) * (1 - F.sigmoid(concept_predicted)))[:, :, None]
        X = cosine_similarity(img_grad.mean(0)[None, :], weight_concept)[0]
        return X

    def ranking_weight(self, weight_concept, weight_label, activation=None, prediction_ind=None, ):
        if activation is None:
            A = weight_concept
            B = weight_label[prediction_ind, :]
            X = np.matmul(B[None, :], A.T)[0]
        else:
            X = np.zeros((len(activation), len(weight_concept)))
            B = weight_label[prediction_ind, :]# * activation[i]
            for i in range(len(activation)):
                A = weight_concept * activation[i]
                # X[i] = cosine_similarity(B[None, :], A)
                X[i] = np.matmul(B[None, :], A.T)
            X = X.mean(0)
        return X

    def single_weight_synthesis(self,  component_weights, target_weight):
        w = np.matmul(target_weight, np.linalg.pinv(component_weights))
        # combination_score = cosine_similarity(target_weight[None, :], np.matmul(w, component_weights)[None, :])
        combination_score = 1 - np.linalg.norm(target_weight - np.matmul(w, component_weights)) ** 2
        return w, combination_score
        # X = nn.Parameter(torch.randn((weight_softmax.shape[0], weight_clf.shape[0])))
        # loss = nn.MSELoss()
        # opt = optim.Adam([X], lr=0.02)
        # W = V(torch.from_numpy(weight_concept))
        # Y = V(torch.from_numpy(weight_label))
        # if settings.GPU:
        #     loss, X, W, Y = loss.cuda(), X.cuda(), W.cuda(), Y.cuda()
        # for i in range(5000):
        #     err = loss(torch.matmul(F.leaky_relu(X, 0.005), W), Y)
        #     print("epoch %02d: err %.8f" % (i, err.item()))
        #     opt.zero_grad()
        #     err.backward()
        #     opt.step()
        #
        # X = X.data.numpy()

    def weight_retrieval(self, model, feat_clf, feat_labels=None):
        params = list(model.parameters())
        if settings.GPU:
            weight_softmax = params[-2].data.cpu().numpy()
            weight_clf = feat_clf.fc.weight.data.cpu().numpy()
        else:
            weight_softmax = params[-2].data.numpy()
            weight_clf = feat_clf.fc.weight.data.numpy()
        weight_label = weight_softmax / np.linalg.norm(weight_softmax, axis=1)[:, None]
        weight_concept = weight_clf / np.linalg.norm(weight_clf, axis=1)[:, None]
        # weight_label = weight_softmax
        # weight_concept = weight_clf
        # A = np.maximum(weight_concept,0)
        # B = np.maximum(weight_label,0)[prediction_ind, :]
        A = weight_concept
        B = weight_label
        X = np.matmul(B, A.T)
        if settings.COMPRESSED_INDEX:
            feat_labels = [feat_labels[concept] for concept in feat_clf.valid_concepts]
        if settings.DATASET == "places365":
            model_labels = places365_categories
        elif settings.DATASET == "imagenet":
            model_labels = imagenet_categories
        mat_sort = X.argsort(1)[:, :-6:-1]  # [:, :5]#
        for pi in range(len(X)):
            prediction = model_labels[pi]
            print(prediction, end=":\t")
            concept_inds = mat_sort[pi, :]
            for concept_ind in concept_inds:
                print("%s(%.2f)" % (feat_labels[concept_ind], X[pi, concept_ind]), end=",\t")
            print()

    def concept_indexmap(self, feat, feature_name, save=True):
        b, u, h, w = feat.shape
        print("generating concept index map ...")
        filename = os.path.join(settings.OUTPUT_FOLDER, "%s-concept-map.pickle" % feature_name)
        if os.path.exists(filename):
            with open(filename,'rb') as f:
                return pickle.load(f)

        concept_indexes = [set() for i in range(len(self.data.label))]
        pd = SegmentationPrefetcher(self.data, categories=self.data.category_names(),
                                    once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                    ahead=settings.TALLY_AHEAD)
        for batch in pd.batches():
            for concept_map in batch:
                scalars, pixels = [], []
                for cat in self.data.category_names():
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        pixels.append(label_group)
                for pixel in pixels:
                    pixel = imresize(pixel[0], (h, w), interp='nearest', mode='F').astype(int)
                    for hi in range(h):
                        for wi in range(w):
                            if pixel[hi,wi]:
                                concept_indexes[pixel[hi,wi]].add((concept_map['i'], hi, wi))

        if save:
            with open(filename,'wb') as f:
                pickle.dump(concept_indexes, f)
        return concept_indexes

    def embedding2d_feat(self, feat, feature_name, alg="se", save=True):
        filename = os.path.join(settings.OUTPUT_FOLDER, "%s-%s.pickle" % (feature_name,alg))
        if os.path.exists(filename):
            return np.load(filename)
        b, u, h, w = feat.shape
        feat.transpose(0,2,3,1)
        feat.shape = (b * w * h, u)
        if alg == 'se':
            feat_nse = SpectralEmbedding(n_components=2).fit_transform(feat)
        else:
            feat_nse = TSNE(n_components=2, verbose=2).fit_transform(feat)
        feat_nse.shape = (b, w, h, 2)
        if save:
            np.save(filename,feat_nse)
        return feat_nse

    def cluster(self, feat, feature_name, linkage='ward', save=True):
        filename = os.path.join(settings.OUTPUT_FOLDER, "%s-cluster.npy" % feature_name)
        if os.path.exists(filename):
            return np.load(filename)
        b, u, h, w = feat.shape
        feat.transpose(0,2,3,1)
        feat.shape = (b * w * h, u)
        clustering = AgglomerativeClustering(linkage=linkage, n_clusters=10)
        clustering.fit(feat)
        if save:
            np.save(filename,(clustering.labels_,clustering.children_))
        return clustering.labels_,clustering.children_

    def instance_segment_by_id(self, feat, img_index, feat_clf):
        _, u, h, w = feat.shape
        img_feat = feat[img_index].transpose(1,2,0)
        img_feat.shape = (h * w, u)
        concept_predicted = feat_clf(V(torch.FloatTensor(img_feat), volatile=True))
        concept_predicted = concept_predicted.data.numpy().reshape(h, w, -1)
        img = imread(os.path.join(settings.DATA_DIRECTORY,'images', self.data.image[img_index]['image']))
        imsave(os.path.join(settings.OUTPUT_FOLDER, 'original.jpg'), img)
        return concept_predicted.argmax(2)

    def cam_mat(self, mat, above_zero=False):
        if above_zero:
            mat = np.maximum(mat, 0)
        if len(mat.shape) == 3:
            mat = mat.sum(2)
        mat = mat - np.min(mat)
        mat = mat / np.max(mat)
        return mat

    def instance_segment_by_file(self, model, image_file, feat_clf):
        #feature extraction
        org_img = imread(image_file)
        org_img = imresize(org_img, (settings.IMG_SIZE, settings.IMG_SIZE))
        if org_img.shape.__len__() == 2:
            org_img = org_img[:,:,None].repeat(3,axis=2)
        img_feat, img_grad, prediction_ind, prediction = self.single_feature_extraction(model, org_img)

        # feature classification
        h, w, u = img_feat.shape
        seg_resolution = h
        img_feat_resized = np.zeros((seg_resolution, seg_resolution, u))
        for i in range(u):
            img_feat_resized[:,:,i] = imresize(img_feat[:,:,i], (seg_resolution, seg_resolution), mode="F")
        img_feat_resized.shape = (seg_resolution * seg_resolution, u)
        concept_predicted = feat_clf(V(torch.FloatTensor(img_feat_resized), volatile=True))
        concept_predicted = concept_predicted.data.numpy().reshape(seg_resolution, seg_resolution, -1)
        concept_inds = concept_predicted.argmax(2)
        concept_colors = np.array(random_color(concept_predicted.shape[2])) * 256

        # feature visualization
        vis_size = settings.IMG_SIZE * 2
        cam_mat = self.cam_mat(img_feat * img_grad, above_zero=True)
        cam_mask = 255 * imresize(cam_mat, (settings.IMG_SIZE, settings.IMG_SIZE), mode="F")
        cam_mask = cv2.applyColorMap(np.uint8(cam_mask), cv2.COLORMAP_JET)[:,:,::-1]
        vis_cam = cam_mask * 0.5 + org_img * 0.5
        vis_cam = Image.fromarray(vis_cam.astype(np.uint8))
        vis_cam = vis_cam.resize((vis_size, vis_size), resample=Image.BILINEAR)

        seg_mask = imresize(concept_colors[concept_inds], (settings.IMG_SIZE, settings.IMG_SIZE), interp='nearest', mode="RGB")
        vis_seg = seg_mask * 0.7 + org_img * 0.3
        vis_seg = Image.fromarray(vis_seg.astype(np.uint8))
        vis_seg = vis_seg.resize((vis_size, vis_size), resample=Image.NEAREST)

        label_seg(vis_seg, vis_size / h, self.data.label, concept_inds, cam=cam_mat)

        vis_img = Image.fromarray(org_img).resize((vis_size, vis_size), resample=Image.BILINEAR)
        vis = imconcat([vis_img, vis_cam, vis_seg], vis_size, vis_size)
        return vis, prediction

        # Y, X = np.meshgrid(np.arange(h), np.arange(w))
        # concept_value = concept_predicted[X, Y, concept_inds]
        # concept_scores = np.exp(concept_value) / np.sum(np.exp(concept_predicted), 2)
        # mask = np.concatenate([concept_colors[concept_inds], concept_scores[:, :, None] * 256], 2)

    def instance_cam_by_file(self, model, image_file, feat_clf, other_params=None, fig_style=0):

        # feature extraction
        org_img = imread(image_file)
        org_img = imresize(org_img, (settings.IMG_SIZE, settings.IMG_SIZE))
        if org_img.shape.__len__() == 2:
            org_img = org_img[:, :, None].repeat(3, axis=2)
        if settings.APP == "vqa":
            img_feat, img_grad, prediction_ind, _ = self.vqa_feature_extraction(model, org_img, *other_params)
            prediction = prediction_ind
        elif settings.APP == "imagecap":
            img_feat, img_grad, prediction_ind, prediction = self.imagecap_feature_extraction(model, org_img)
            prediction = (np.array(model.vocab)[prediction], prediction_ind)
        else:
            img_feat, img_grad, prediction_ind, prediction = self.single_feature_extraction(model, org_img)
        if settings.COMPRESSED_INDEX:
            try:
                labels = [self.data.label[concept] for concept in feat_clf.valid_concepts]
            except Exception:
                labels = [self.data.label[concept] for concept in np.load('cache/valid_concept.npy')]

        else:
            labels = self.data.label
        h, w, u = img_feat.shape

        # feature classification
        seg_resolution = settings.SEG_RESOLUTION
        img_feat_resized = np.zeros((seg_resolution, seg_resolution, u))
        for i in range(u):
            img_feat_resized[:, :, i] = imresize(img_feat[:, :, i], (seg_resolution, seg_resolution), mode="F")
        img_feat_resized.shape = (seg_resolution * seg_resolution, u)

        concept_predicted = feat_clf.fc(V(torch.FloatTensor(img_feat_resized)))
        concept_predicted = concept_predicted.data.numpy().reshape(seg_resolution, seg_resolution, -1)
        # concept_predicted_reg = (concept_predicted - np.min(concept_predicted, 2, keepdims=True)) / np.max(
        #     concept_predicted, 2, keepdims=True)

        concept_inds = concept_predicted.argmax(2)
        concept_colors = np.array(random_color(concept_predicted.shape[2])) * 256

        # feature visualization
        vis_size = settings.IMG_SIZE
        margin = int(vis_size/30)
        img_cam = self.cam_mat(img_feat * img_grad.mean((0,1))[None,None,:], above_zero=False)
        img_camp = self.cam_mat(img_feat * img_grad, above_zero=True)
        vis_cam = vis_cam_mask(img_cam, org_img, vis_size)
        vis_camp = vis_cam_mask(img_camp, org_img, vis_size)

        CONCEPT_CAM_TOPN = settings.BASIS_NUM
        CONCEPT_CAM_BOTTOMN = 0

        if settings.GRAD_CAM:
            weight_clf = feat_clf.fc.weight.data.numpy()
            weight_concept = weight_clf#np.maximum(weight_clf, 0)
            weight_concept = weight_concept / np.linalg.norm(weight_concept, axis=1)[:, None]
            # ranking = self.ranking_gradient(weight_concept, img_grad.reshape(-1, u))
            # component_weights = weight_concept[ranking.argsort()[:-5 - 1:-1], :]
            target_weight = img_grad.mean((0,1))
            target_weight = target_weight / np.linalg.norm(target_weight)
            # w, combination_score = self.single_weight_synthesis(component_weights, target_weight)
            rankings, scores, coefficients, residuals = self.decompose_Gram_Schmidt(weight_concept, target_weight[None, :], MAX=settings.BASIS_NUM)
            ranking = rankings[0]
            residual = residuals[0]
            d_e = np.linalg.norm(residuals[0]) ** 2

            component_weights = np.vstack([coefficients[0][:settings.BASIS_NUM, None] * weight_concept[ranking], residual[None,:]])
            a = img_feat.mean((0,1))
            a /= np.linalg.norm(a)
            qcas = np.dot(component_weights, a)
            combination_score = sum(abs(qcas))
            inds = qcas[:-1].argsort()[:-CONCEPT_CAM_TOPN - 1:-1]
            concept_masks_ind = ranking[inds]
            scores_topn = coefficients[0][inds]
            contribution = qcas[inds]
        else:
            # activation=img_feat[(img_cam > 0.6).nonzero()]
            weight_label, weight_concept = self.weight_extraction(model, feat_clf)
            # ranking = 1 / (1 + np.exp(-np.matmul(weight_concept, img_feat.reshape(-1,u).transpose()).max(1)))
            # ranking = self.ranking_weight(weight_concept, weight_label, prediction_ind=prediction_ind)
            # component_weights = weight_concept[ranking.argsort()[:-5 - 1:-1], :]
            # concept_masks_ind = ranking.argsort()[:-5 - 1:-1]
            # scores_topn = ranking[concept_masks_ind]

            rankings, errvar, coefficients, residuals_T = np.load(os.path.join(settings.OUTPUT_FOLDER, "decompose.npy"))
            ranking = rankings[prediction_ind].astype(int)
            residual = residuals_T.T[prediction_ind]
            d_e = np.linalg.norm(residual) ** 2
            component_weights = np.vstack([coefficients[prediction_ind][:settings.BASIS_NUM, None] * weight_concept[ranking], residual[None,:]])
            a = img_feat.mean((0,1))
            a /= np.linalg.norm(a)
            qcas = np.dot(component_weights, a)
            combination_score = sum(qcas)
            inds = qcas[:-1].argsort()[:-CONCEPT_CAM_TOPN - 1:-1]
            concept_masks_ind = ranking[inds]
            scores_topn = coefficients[prediction_ind][inds]
            contribution = qcas[inds]

            # target_weight = img_feat.mean((0, 1))# * weight_label[prediction_ind, :]
            # target_weight /= np.linalg.norm(target_weight)
            # # component_weights = img_feat.mean((0, 1))[None,: ] * weight_concept
            # # component_weights /= np.linalg.norm(component_weights, axis=1)[:, None]
            # rankings, scores, ws, residuals = self.decompose_Gram_Schmidt(weight_concept, target_weight[None, :])
            # concept_masks_ind = rankings.ravel()
            # scores_topn = scores.ravel()
            # w = ws.ravel()
            # combination_score = scores_topn[-1]


        concept_masks = concept_predicted[:, :, concept_masks_ind]
        concept_masks = concept_masks * ((scores_topn > 0) * 1)[None, None, :]
        concept_masks = (np.maximum(concept_masks, 0)) / np.max(concept_masks)

        vis_concept_cam = []
        # acc = np.memmap(os.path.join(settings.OUTPUT_FOLDER, "mAP_table.mmap"), dtype=np.float16, mode='r', shape=(660,15))[:, 6][concept_masks_ind]
        # captions = [labels[concept_masks_ind[i]]['name'] + "(%.3f)" % (scores_topn[i]) for i in range(CONCEPT_CAM_TOPN+CONCEPT_CAM_BOTTOMN)]
        for i in range(CONCEPT_CAM_TOPN+CONCEPT_CAM_BOTTOMN):
            vis_concept_cam.append(vis_cam_mask(concept_masks[:,:,i], org_img, vis_size, font_text=None))
        # vis_concept_cam.append(vis_cam_mask(self.cam_mat(np.dot(img_feat, residual)) * 0.8, org_img, vis_size, font_text=None))

        # test = np.matmul(score_mat[prediction_ind], concept_predicted.reshape(w * h, -1).T).reshape(h, w)
        # vis_concept_cam.append(vis_cam_mask(self.cam_mat(test, above_zero=True), org_img, vis_size))

        # seg_mask = imresize(concept_colors[concept_inds], (settings.IMG_SIZE, settings.IMG_SIZE), interp='nearest', mode="RGB")
        # vis_seg = seg_mask * 0.7 + org_img * 0.3
        # vis_seg = Image.fromarray(vis_seg.astype(np.uint8))
        # vis_seg = vis_seg.resize((vis_size, vis_size), resample=Image.NEAREST)
        # label_seg(vis_seg, vis_size, labels, concept_inds, cam=img_cam)
        if fig_style == 0:
            vis_img = Image.fromarray(org_img).resize((vis_size, vis_size), resample=Image.BILINEAR)
            vis = imconcat([vis_img, vis_cam, vis_camp] + vis_concept_cam, vis_size, vis_size, margin=margin)
            captions = ["{%s}: s(%.2f)->%4.2f%%" % (labels[concept_masks_ind[i]]['name'], scores_topn[i], contribution[i] * 100 / combination_score) for i in range(CONCEPT_CAM_TOPN+CONCEPT_CAM_BOTTOMN)]
            captions = ["score {%.2f} residual {de %.2f/(%4.2f%%)}" % (combination_score, d_e, qcas[-1] * 100 / combination_score), "CAM", "CAM+"] + captions
            # captions = ["%.2f * {%s}" % (w[i], captions[i]) for i in range(len(captions))]
            # captions = ["original image", "CAM or grad CAM", "VIS+", "score {%.2f}" % (combination_score)] + captions
            vis_headline = headline(captions, vis_size, vis.height // 4, vis.width, margin=margin)
            vis = imstack([vis_headline, vis])
        elif fig_style == 1:

            vis_img = Image.fromarray(org_img).resize((vis_size, vis_size), resample=Image.BILINEAR)
            vis_bm = big_margin(vis_size)
            vis = imconcat([vis_img, vis_cam, vis_bm] + vis_concept_cam[:3], vis_size, vis_size, margin=margin)
            captions = ["%s(%4.2f%%)" % (labels[concept_masks_ind[i]]['name'], contribution[i] * 100 / combination_score) for i in range(3)]
            captions = ["%s(%.2f) " % (prediction, combination_score)] + captions
            vis_headline = headline2(captions, vis_size, vis.height // 5, vis.width, margin=margin)
            vis = imstack([vis_headline, vis])
        return vis, prediction

