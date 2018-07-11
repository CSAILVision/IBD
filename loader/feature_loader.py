import torch
import numpy as np
import random
import os
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from loader.data_loader import SegmentationPrefetcher
from scipy.misc import imresize
import settings
import math

def nonzero_filter(batch):
    # ret = list(filter(lambda b: b[1] != 0, batch))
    feat = np.concatenate([x for x,y in batch])
    label = np.concatenate([y for x,y in batch]).ravel()
    if feat.__len__() == 0:
        return (0,0)
    return torch.FloatTensor(feat), torch.LongTensor(label)

class FeatureDataset(Dataset):
    def generate_label(self, shape, seg_data):
        b, u, h, w = shape
        img_ind = 0
        print("generating concept index map ...")
        pd = SegmentationPrefetcher(seg_data, categories=seg_data.category_names(),
                                    once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                    ahead=settings.TALLY_AHEAD)
        for batch in pd.batches():
            print("handling image index %d" % img_ind)
            for concept_map in batch:
                scalars, pixels = [], []
                for cat in seg_data.category_names():
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        pixels.append(label_group)

                for i, pixel in enumerate(pixels):
                    pixels[i] = imresize(pixel[0], (settings.SEG_RESOLUTION, settings.SEG_RESOLUTION), interp='nearest', mode='F').astype(int)

                labels = np.array(pixels)
                if len(labels) == 2:
                    labels = labels[0] + (labels[0] == 0) * labels[1]
                else:
                    labels = labels[0]

                self.labels[img_ind] = labels
                img_ind += 1


    @staticmethod
    def feat_resize_func(size):
        def feat_resize(mat):
            return imresize(mat.reshape(size), (settings.SEG_RESOLUTION, settings.SEG_RESOLUTION), interp='bilinear', mode='F')
        return feat_resize

    def __init__(self, feat, feat_name, seg_data, concept_size):
        b, u, h, w = feat.shape
        feat = feat.transpose(0,2,3,1)
        self.dims = (b, settings.SEG_RESOLUTION, settings.SEG_RESOLUTION)
        self.feat = feat
        self.labels = None
        self.label_size = concept_size
        filename = os.path.join(settings.OUTPUT_FOLDER, "%s-concept-pixels.mmap" % feat_name)
        if os.path.exists(filename):
            print("loading concept index map ...")
            # self.labels = np.zeros(self.dims)
            self.labels = np.memmap(filename, dtype=float, mode='r', shape=self.dims)
        else:
            self.labels = np.memmap(filename, dtype=float, mode='w+', shape=self.dims)
            self.generate_label((b, u, h, w), seg_data)

    def __getitem__(self, img_id):
        h = self.feat[img_id].shape[0]
        x = np.apply_along_axis(FeatureDataset.feat_resize_func((h,h)), 0, self.feat[img_id].reshape((h*h, -1)))
        labels = self.labels[img_id]
        labels = labels.reshape(settings.SEG_RESOLUTION ** 2, -1)
        x = x.reshape(settings.SEG_RESOLUTION ** 2, -1)
        x = x[labels.ravel() != 0]
        labels = labels[labels.ravel() != 0]
        return x, labels

    # def __getitem__(self, seq_id):
    #     index = np.unravel_index(seq_id, self.dims)
    #     x = torch.Tensor(self.feat[index])
    #     if self.labels[index[0]] is None or len(self.labels[index[0]]) == 0:
    #         y = 0
    #     else:
    #         y = self.labels[index[0]][0][index[1]][index[2]]
    #         if y == 0 and len(self.labels[index[0]][0]) == 2:
    #             y = self.labels[index[0]][1][index[1]][index[2]]
    #     return x, y

    def __len__(self):
        return self.dims[0]# * self.dims[1] * self.dims[2]

class ConceptDataset(Dataset):
    def __init__(self, feat, feat_name, seg_data, concept_size):
        b, u, h, w = feat.shape
        feat = feat.transpose(0,2,3,1)
        self.dims = (b, w, h)
        self.feat = feat
        self.labels = [None] * b
        self.concept_indexes = [set() for _ in range(concept_size)]
        self.label_size = concept_size
        self.generate_label((b, u, h, w), feat_name, seg_data)
        self.concept_count = np.zeros(len(seg_data.label), dtype=np.int32)
        for label in self.labels:
            self.concept_count += np.bincount(label.ravel(), minlength=len(seg_data.label))
        self.concept_count[0] = 0

    def generate_label(self, shape, feature_name, seg_data):
        b, u, h, w = shape
        img_ind = 0
        filename = os.path.join(settings.OUTPUT_FOLDER, "%s-concept-map.npy" % feature_name)
        if os.path.exists(filename):
            print("loading concept index map ...")
            self.concept_indexes, self.labels = np.load(filename)
            return

        print("generating concept index map ...")
        pd = SegmentationPrefetcher(seg_data, categories=seg_data.category_names(),
                                    once=True, batch_size=settings.TALLY_BATCH_SIZE,
                                    ahead=settings.TALLY_AHEAD)
        for batch in pd.batches():
            print("handling image index %d" % img_ind)
            for concept_map in batch:
                scalars, pixels = [], []
                for cat in seg_data.category_names():
                    label_group = concept_map[cat]
                    shape = np.shape(label_group)
                    if len(shape) % 2 == 0:
                        label_group = [label_group]
                    if len(shape) < 2:
                        scalars += label_group
                    else:
                        pixels.append(label_group)

                if settings.SINGLE_LABEL:
                    if pixels:
                        pixel = imresize(pixels[0][0], (h, w), interp='nearest', mode='F').astype(int)
                        self.labels[img_ind] = pixel
                        for hi in range(h):
                            for wi in range(w):
                                if pixel[hi, wi]:
                                    self.concept_indexes[pixel[hi, wi]].add((img_ind, hi, wi))
                    else:
                        self.labels[img_ind] = np.full((h, w), scalars[0])
                        for hi in range(h):
                            for wi in range(w):
                                if pixel[hi, wi]:
                                    self.concept_indexes[scalars[0]].add((img_ind, hi, wi))

                else:
                    if len(pixels) >= 1:
                        pixels = np.concatenate(pixels, 0)
                        pixel_sm = np.zeros((len(pixels), h, w), dtype=np.int16)
                        for i in range(len(pixels)):
                            # ty, tx = (np.arange(ts) for ts in (h, w))
                            # sy, sx = (np.arange(ss) * s + o for ss, s, o in zip(pixels[i].shape, (4, 4), (5, 5)))
                            # from scipy.interpolate import RectBivariateSpline
                            # levels = RectBivariateSpline(sy, sx, pixels[i], kx=1, ky=1)(ty, tx, grid=True)
                            pixel_sm[i] = imresize(pixels[i], (h, w), interp='nearest', mode='F').astype(int)
                            for hi in range(h):
                                for wi in range(w):
                                    if pixel_sm[i][hi, wi]:
                                        self.concept_indexes[pixel_sm[i][hi, wi]].add((img_ind, hi, wi))
                        self.labels[img_ind] = pixel_sm
                img_ind += 1

        for label in range(self.label_size):
            self.concept_indexes[label] = np.array(list(self.concept_indexes[label]))
        np.save(filename, (self.concept_indexes, self.labels))

    def getitem_by_seqind(self, index):
        x = torch.Tensor(self.feat[index])
        y = torch.zeros(self.label_size)
        if self.labels[index[0]] is not None:
            y_ = self.labels[index[0]][:, index[1], index[2]]
            if y_.max() != 0:
                if settings.SINGLE_LABEL:
                    y[y_[0]] = 1.0
                else:
                    y[torch.LongTensor(y_[y_ > 0])] = 1.0
        return x, y

    def __getitem__(self, concept_ind):
        seq_ind = self.concept_indexes[concept_ind][np.random.randint(0, len(self.concept_indexes[concept_ind]))]
        x = torch.Tensor(self.feat[tuple(seq_ind)])
        # y = torch.zeros(self.label_size)
        # y[concept_ind] = 1.0
        y = concept_ind
        return x, y

    def __len__(self):
        return self.dims[0] * self.dims[1] * self.dims[2]

class ConceptSampler(Sampler):

    def __init__(self, concept_count, samples, selection=None):
        self.concept_count = concept_count
        self.samples = len(concept_count) * samples
        self.selection = selection

    def __iter__(self):
        distribute = np.asarray(self.concept_count > 0, np.float)
        if self.selection is not None:
            selection = np.zeros(len(self.concept_count))
            selection[self.selection] = 1
            distribute *= selection
        distribute[0] = 0
        return iter(torch.multinomial(torch.FloatTensor(distribute), self.samples, replacement=True))

    def __len__(self):
        return self.samples

class SingleConceptDataset(Dataset):
    def __init__(self, feat, raw_data):
        self.feat = feat
        b, h, w, u= feat.shape
        self.dim = (b,h,w)
        self.x, self.y = raw_data

    def __getitem__(self, i):
        ind = tuple(self.x[i])
        x = torch.Tensor(self.feat[ind])
        y = self.y[i]
        return np.ravel_multi_index(ind, self.dim), x, y


    def __len__(self):
        return len(self.x)

class concept_loader_factory:
    def __init__(self, feat, feat_name, seg_data, concept_size, concept_dataset=None, ):
        super(concept_loader_factory, self).__init__()
        if concept_dataset is not None:
            self.dataset = concept_dataset
        else:
            self.dataset = ConceptDataset(feat, feat_name, seg_data, concept_size)
        feat = feat.transpose(0, 2, 3, 1)
        self.feature = feat
        self.feature_name = feat_name
        self.seg_data = seg_data
        self.concept_size = concept_size

    def negative_mining_loader(self, split_ratio=3, sample_ratio=2, shuffle=True, neg_scores=None):
        b, h, w, u = self.feature.shape
        valid_concepts = self.dataset.concept_count.nonzero()[0]
        single_concept_train_loaders = [None] * len(valid_concepts)
        single_concept_val_loaders = [None] * len(valid_concepts)
        # valid_concepts = [12]
        for count, concept in enumerate(valid_concepts):
            print("building concept loader for concept %d/%d" % (concept, self.concept_size))
            positive_samples = self.dataset.concept_indexes[concept]
            cache_name = os.path.join(settings.OUTPUT_FOLDER, 'sample_cache', '%d-neg.npy' % concept)
            negative_samples = np.load(cache_name)
            if neg_scores is None:
                sample_ind = np.random.choice(len(negative_samples), len(positive_samples) * sample_ratio, replace=False)
            else:
                sample_ind = neg_scores[count].argsort()[:-len(positive_samples) * sample_ratio - 1:-1]
            negative_samples = negative_samples[sample_ind, :]
            train_dataset, val_dataset = self._dataset_maker(negative_samples, positive_samples, split_ratio=split_ratio)
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle,
                                      batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS)
            single_concept_train_loaders[count] = train_loader
            if val_dataset is not None:
                val_loader = DataLoader(dataset=val_dataset, shuffle=shuffle,
                                        batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS)
                single_concept_val_loaders[count] = val_loader
        return single_concept_train_loaders, single_concept_val_loaders

    def negative_test_concept_loader(self, verbose=True, sample_ratio=20):
        b, h, w, u = self.feature.shape
        valid_concepts = self.dataset.concept_count.nonzero()[0]
        negative_loaders = [None] * len(valid_concepts)
        for count, concept in enumerate(valid_concepts):

            print("building concept loader for concept %d/%d" % (concept, self.concept_size))
            positive_samples = self.dataset.concept_indexes[concept]
            cache_name = os.path.join(settings.OUTPUT_FOLDER, 'sample_cache', '%d-neg.npy' % concept)
            if os.path.exists(cache_name) and not verbose:
                negative_samples = np.load(cache_name)
            else:
                #exclude positive samples
                unallowed_values = np.apply_along_axis(lambda ind: np.ravel_multi_index(ind, (b, h, w)), 1, positive_samples)
                allowed_values = np.delete(np.arange(b * h * w), unallowed_values, None)
                negative_samples = np.random.choice(allowed_values, min(len(allowed_values), sample_ratio * len(positive_samples)), replace=False)
                negative_samples = np.apply_along_axis(lambda ind: np.unravel_index(ind, (b, h, w)), 0, negative_samples).T
                np.save(cache_name, negative_samples)
            t_n_x = negative_samples
            t_n_y = np.zeros(len(t_n_x), dtype=np.float32)
            neg_test_loader = DataLoader(dataset=SingleConceptDataset(self.feature, (t_n_x, t_n_y)), shuffle=False,
                                      batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS)
            negative_loaders[count] = neg_test_loader
        return negative_loaders

    def fixed_val_loader(self, sample_ratio=20, verbose=False, shuffle=False):
        b, h, w, u = self.feature.shape
        valid_concepts = self.dataset.concept_count.nonzero()[0]
        fix_val_loaders = [None] * len(valid_concepts)
        for count, concept in enumerate(valid_concepts):

            print("building concept loader for concept %d/%d" % (concept, self.concept_size))
            positive_samples = self.dataset.concept_indexes[concept]
            cache_name = os.path.join(settings.CACHE_PATH, '%d-neg.npy' % concept)
            if os.path.exists(cache_name) and not verbose:
                negative_samples = np.load(cache_name)
            else:
                # exclude positive samples0
                unallowed_values = np.apply_along_axis(lambda ind: np.ravel_multi_index(ind, (b, h, w)), 1,
                                                       positive_samples)
                allowed_values = np.delete(np.arange(b * h * w), unallowed_values, None)
                negative_samples = np.random.choice(allowed_values,
                                                    min(len(allowed_values), sample_ratio * len(positive_samples)),
                                                    replace=False)
                negative_samples = np.apply_along_axis(lambda ind: np.unravel_index(ind, (b, h, w)), 0,
                                                       negative_samples).T
                np.save(cache_name, negative_samples)

            val_dataset, _ = self._dataset_maker(negative_samples, positive_samples, split_ratio=None)
            val_loader = DataLoader(dataset=val_dataset, shuffle=shuffle,
                                      batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS)
            fix_val_loaders[count] = val_loader
        return fix_val_loaders

    def random_concept_loader(self, split_ratio=3, sample_ratio=3, verbose=True, shuffle=True):
        b, h, w, u = self.feature.shape
        valid_concepts = self.dataset.concept_count.nonzero()[0]
        single_concept_train_loaders = [None] * len(valid_concepts)
        single_concept_val_loaders = [None] * len(valid_concepts)
        for count, concept in enumerate(valid_concepts):

            print("building concept loader for concept %d/%d" % (concept, self.concept_size))
            positive_samples = self.dataset.concept_indexes[concept]
            cache_name = os.path.join(settings.OUTPUT_FOLDER, 'sample_cache', '%d-neg.npy' % concept)
            if os.path.exists(cache_name) and not verbose:
                negative_samples = np.load(cache_name)
            else:
                #exclude positive samples
                unallowed_values = np.apply_along_axis(lambda ind: np.ravel_multi_index(ind, (b, h, w)), 1, positive_samples)
                allowed_values = np.delete(np.arange(b * h * w), unallowed_values, None)
                negative_samples = np.random.choice(allowed_values, min(len(allowed_values), sample_ratio * len(positive_samples)), replace=False)
                negative_samples = np.apply_along_axis(lambda ind: np.unravel_index(ind, (b, h, w)), 0, negative_samples).T
                np.save(cache_name, negative_samples)

            train_dataset, val_dataset = self._dataset_maker(negative_samples, positive_samples, split_ratio=split_ratio)
            train_loader = DataLoader(dataset=train_dataset, shuffle=shuffle,
                                      batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS)
            single_concept_train_loaders[count] = train_loader
            if val_dataset is not None:
                val_loader = DataLoader(dataset=val_dataset, shuffle=shuffle,
                                        batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS)
                single_concept_val_loaders[count] = val_loader


        return single_concept_train_loaders, single_concept_val_loaders

    def _dataset_maker(self, negative_samples, positive_samples, split_ratio=None):
        if split_ratio is not None:
            p_split_ind = len(positive_samples) * split_ratio // (split_ratio + 1)
            t_p_x = positive_samples[:p_split_ind, :]
            v_p_x = positive_samples[p_split_ind:, :]
            t_p_y = np.ones(len(t_p_x), dtype=np.float32)
            v_p_y = np.ones(len(v_p_x), dtype=np.float32)

            n_split_ind = len(negative_samples) * split_ratio // (split_ratio + 1)
            t_n_x = negative_samples[:n_split_ind, :]
            v_n_x = negative_samples[n_split_ind:, :]
            t_n_y = np.zeros(len(t_n_x), dtype=np.float32)
            v_n_y = np.zeros(len(v_n_x), dtype=np.float32)

            t_x = np.concatenate([t_p_x, t_n_x], 0)
            t_y = np.concatenate([t_p_y, t_n_y], 0)
            v_x = np.concatenate([v_p_x, v_n_x], 0)
            v_y = np.concatenate([v_p_y, v_n_y], 0)
            return SingleConceptDataset(self.feature, (t_x, t_y)), SingleConceptDataset(self.feature, (v_x, v_y))
        else:
            t_p_x = positive_samples
            t_p_y = np.ones(len(t_p_x), dtype=np.float32)
            t_n_x = negative_samples
            t_n_y = np.zeros(len(t_n_x), dtype=np.float32)
            t_x = np.concatenate([t_p_x, t_n_x], 0)
            t_y = np.concatenate([t_p_y, t_n_y], 0)
            return SingleConceptDataset(self.feature, (t_x, t_y)), None





def concept_loader(feat, feat_name, seg_data, concept_size, split=False):
    dataset = ConceptDataset(feat, feat_name, seg_data, concept_size)
    train_loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE,
                        sampler=ConceptSampler(dataset.concept_count, 1000), num_workers=settings.WORKERS)
    if split:
        val_loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE,
                        sampler=ConceptSampler(dataset.concept_count, 20), num_workers=settings.WORKERS)
    else:
        val_loader = None
    return train_loader, val_loader

def feature_loader(feat, feat_name, seg_data, concept_size, split=False):
    dataset = FeatureDataset(feat, feat_name, seg_data, concept_size)
    if split:
        test_idx, train_idx = range(0, len(dataset)//4), range(len(dataset)//4, len(dataset))
        train_loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE, sampler=SubsetRandomSampler(train_idx), num_workers=settings.WORKERS, collate_fn=nonzero_filter)
        test_loader = DataLoader(dataset=dataset, batch_size=settings.BATCH_SIZE, sampler=SubsetRandomSampler(test_idx), num_workers=settings.WORKERS, collate_fn=nonzero_filter)
        return train_loader, test_loader
    else:
        loader = DataLoader(dataset=dataset,batch_size=settings.BATCH_SIZE, num_workers=settings.WORKERS, collate_fn=nonzero_filter)
        return loader, None
