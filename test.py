
import os
import settings
from loader.model_loader import loadmodel
from util.feature_operation import FeatureOperator
from util.clean import clean
from util.feature_decoder import SingleSigmoidFeatureClassifier
from util.image_operation import *
from PIL import Image
import numpy as np
from scipy.misc import imresize, imread
from visualize.plot import random_color
from torch.autograd import Variable as V
import torch


model = loadmodel()
fo = FeatureOperator()

features, _ = fo.feature_extraction(model=model)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    feat_clf = SingleSigmoidFeatureClassifier(feature=features[layer_id], layer=layer, fo=fo)
    feat_clf.load_snapshot(14, unbiased=True)

    if not settings.GRAD_CAM:
        fo.weight_decompose(model, feat_clf, feat_labels=[l['name'] for l in fo.data.label])

    with open(settings.DATASET_INDEX_FILE) as f:
        image_list = f.readlines()
        predictions = []
        outpath = os.path.join(settings.OUTPUT_FOLDER, 'html', 'image')
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        for image_ind, file in enumerate(image_list):
            print("generating figure on %03d" % image_ind)
            image_file = os.path.join(settings.DATASET_PATH, file.strip())

            # feature extraction
            org_img = imread(image_file)
            org_img = imresize(org_img, (settings.IMG_SIZE, settings.IMG_SIZE))
            if org_img.shape.__len__() == 2:
                org_img = org_img[:, :, None].repeat(3, axis=2)
            img_feat, img_grad, prediction_ind, prediction = fo.single_feature_extraction(model, org_img)
            if settings.COMPRESSED_INDEX:
                try:
                    labels = [fo.data.label[concept] for concept in feat_clf.valid_concepts]
                except Exception:
                    labels = [fo.data.label[concept] for concept in np.load('cache/valid_concept.npy')]

            else:
                labels = fo.data.label
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
            margin = int(vis_size / 30)
            img_cam = fo.cam_mat(img_feat * img_grad.mean((0, 1))[None, None, :], above_zero=False)
            vis_cam = vis_cam_mask(img_cam, org_img, vis_size)
            CONCEPT_CAM_TOPN = settings.BASIS_NUM
            CONCEPT_CAM_BOTTOMN = 0

            if settings.GRAD_CAM:
                weight_clf = feat_clf.fc.weight.data.numpy()
                weight_concept = weight_clf  # np.maximum(weight_clf, 0)
                weight_concept = weight_concept / np.linalg.norm(weight_concept, axis=1)[:, None]
                target_weight = img_grad.mean((0, 1))
                target_weight = target_weight / np.linalg.norm(target_weight)
                rankings, scores, coefficients, residuals = fo.decompose_Gram_Schmidt(weight_concept,
                                                                                        target_weight[None, :],
                                                                                        MAX=settings.BASIS_NUM)
                ranking = rankings[0]
                residual = residuals[0]
                d_e = np.linalg.norm(residuals[0]) ** 2

                component_weights = np.vstack(
                    [coefficients[0][:settings.BASIS_NUM, None] * weight_concept[ranking], residual[None, :]])
                a = img_feat.mean((0, 1))
                a /= np.linalg.norm(a)
                qcas = np.dot(component_weights, a)
                combination_score = sum(abs(qcas))
                inds = qcas[:-1].argsort()[:-CONCEPT_CAM_TOPN - 1:-1]
                concept_masks_ind = ranking[inds]
                scores_topn = coefficients[0][inds]
                contribution = qcas[inds]
            else:
                weight_label, weight_concept = fo.weight_extraction(model, feat_clf)

                rankings, errvar, coefficients, residuals_T = np.load(
                    os.path.join(settings.OUTPUT_FOLDER, "decompose.npy"))
                ranking = rankings[prediction_ind].astype(int)
                residual = residuals_T.T[prediction_ind]
                d_e = np.linalg.norm(residual) ** 2
                component_weights = np.vstack(
                    [coefficients[prediction_ind][:settings.BASIS_NUM, None] * weight_concept[ranking],
                     residual[None, :]])
                a = img_feat.mean((0, 1))
                a /= np.linalg.norm(a)
                qcas = np.dot(component_weights, a)
                combination_score = sum(qcas)
                inds = qcas[:-1].argsort()[:-CONCEPT_CAM_TOPN - 1:-1]
                concept_masks_ind = ranking[inds]
                scores_topn = coefficients[prediction_ind][inds]
                contribution = qcas[inds]


            concept_masks = concept_predicted[:, :, concept_masks_ind]
            concept_masks = concept_masks * ((scores_topn > 0) * 1)[None, None, :]
            concept_masks = (np.maximum(concept_masks, 0)) / np.max(concept_masks)

            vis_concept_cam = []
            for i in range(CONCEPT_CAM_TOPN + CONCEPT_CAM_BOTTOMN):
                vis_concept_cam.append(vis_cam_mask(concept_masks[:, :, i], org_img, vis_size, font_text=None))

            vis_img = Image.fromarray(org_img).resize((vis_size, vis_size), resample=Image.BILINEAR)
            vis_bm = big_margin(vis_size)
            vis = imconcat([vis_img, vis_cam, vis_bm] + vis_concept_cam[:3], vis_size, vis_size, margin=margin)
            captions = [
                "%s(%4.2f%%)" % (labels[concept_masks_ind[i]]['name'], contribution[i] * 100 / combination_score)
                for i in range(3)]
            captions = ["%s(%.2f) " % (prediction, combination_score)] + captions
            vis_headline = headline2(captions, vis_size, vis.height // 5, vis.width, margin=margin)
            vis = imstack([vis_headline, vis])

            predictions.append(prediction)
            vis.save(os.path.join(outpath, "%03d.jpg" % image_ind))

    f = open(os.path.join(settings.OUTPUT_FOLDER, 'html', 'result.html'), 'w')
    f.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n<title>\n</title>\n</head>\n<body>\n")
    for ind in range(len(predictions)):
        headline = "<h3>%d  %s</h2><br>\n" % (ind, predictions[ind])
        imageline = "<img id='%d' height='%d' src='%s'>\n" % (ind, settings.IMG_SIZE, os.path.join('image', '%03d.jpg' % ind))
        f.write(headline)
        f.write(imageline)
    f.write("</body>\n</html>\n")
    f.close()


if settings.CLEAN:
    clean()


