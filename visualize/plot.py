
import os
import torch
import colorsys
import settings
import numpy as np
from sklearn.manifold import TSNE, SpectralEmbedding
from loader.feature_loader import concept_loader
import matplotlib.pyplot as plt


def random_color(labels, type="soft"):
    if type == "soft":
        HSVcolors = [(np.random.uniform(low=0.1, high=0.9),
                      np.random.uniform(low=0.2, high=0.6),
                      np.random.uniform(low=0.6, high=0.95)) for _ in range(labels)]
    elif type == "bright":
        HSVcolors = [(np.random.uniform(low=0.0, high=1),
                      np.random.uniform(low=0.2, high=1),
                      np.random.uniform(low=0.9, high=1)) for _ in range(labels)]
    return [colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]) for HSVcolor in HSVcolors]


# Generate random colormap
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """

    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

def embedding_map_sample(feat, layer, fo):
    sample_size = 1000
    feat_loader_train, feat_loader_test = concept_loader(feat, layer, fo.data, len(fo.data.label), split=True, batch_size=sample_size)
    from sklearn.manifold import TSNE, SpectralEmbedding
    feat, label = feat_loader_test.__iter__().next()
    feat_nse = TSNE(n_components=2, verbose=2).fit_transform(feat.numpy())
    # feat_nse = SpectralEmbedding(n_components=2).fit_transform(feat)
    HSVcolors = [(np.random.uniform(low=0.0, high=1),
                  np.random.uniform(low=0.2, high=1),
                  np.random.uniform(low=0.9, high=1)) for i in range(label.max() + 1)]
    RGBcolors = np.array([colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]) for HSVcolor in HSVcolors])
    plt.scatter(feat_nse[:,0], feat_nse[:,1], c=RGBcolors[label.numpy()], alpha=.5)
    plt.show()


def embedding_map(feat2d, c_map, labelcat):

    marker_cat = ["x", "o", "v", "+", "s", "*"]
    HSVcolors = [(np.random.uniform(low=0.0, high=1),
                  np.random.uniform(low=0.2, high=1),
                  np.random.uniform(low=0.9, high=1)) for i in range(len(c_map))]
    RGBcolors = [colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]) for HSVcolor in HSVcolors]

    for cid,ind_set in enumerate(c_map):
        if ind_set.__len__() == 0:
            continue
        ind_set = np.array(list(ind_set))
        cat = labelcat[cid].nonzero()[0][0]
        points = feat2d[ind_set[:, 0], ind_set[:, 1], ind_set[:, 2]]
        plt.scatter(points[:,0], points[:,1], c=RGBcolors[cid], marker=marker_cat[cat], alpha=.5)

    plt.savefig(os.path.join(settings.OUTPUT_FOLDER,'image','feat2d.jpg'))


#weight_vis(model, imagenet_categories.values(), feat_clf, [l['name'] for l in fo.data.label], save_path='cachecache/nse.npy')

def weight_vis(model, model_labels, feat_clf, feat_clf_labels, save_path=None):
    params = list(model.parameters())
    weight_softmax = params[-2].data.numpy()
    if settings.GPU:
        weight_clf = feat_clf.fc.weight.data.cpu().numpy()
    else:
        weight_clf = feat_clf.fc.weight.data.numpy()

    if save_path and os.path.exists(save_path):
        nse = np.load(save_path)
    else:
        nse = TSNE(n_components=2, verbose=2).fit_transform(np.concatenate([weight_softmax, weight_clf]))
        if save_path:
            np.save(save_path, nse)
    plt.figure()
    plt.scatter(nse[:len(weight_softmax),0],nse[:len(weight_softmax),1], 10, c='r')
    plt.scatter(nse[len(weight_softmax):, 0], nse[len(weight_softmax):, 1], 10, c='b')
    for i,label in enumerate(model_labels):
        plt.text(nse[i,0], nse[i,1], label, fontdict={'size': 6, 'color': 'r'})
    for i,label in enumerate(feat_clf_labels):
        plt.text(nse[i+len(weight_softmax), 0], nse[i+len(weight_softmax), 1], label, fontdict={'size': 6, 'color': 'b'})
    plt.show()

def image_summary(fo, model, feat_clf):
    outpath = os.path.join(settings.OUTPUT_FOLDER, 'html', 'image')
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    if settings.APP == "classification" or settings.APP == "imagecap":
        # image_file ='/home/sunyiyou/PycharmProjects/test/example4.jpg'
        # vis, prediction = fo.instance_cam_by_file(model, image_file, feat_clf)
        # vis.save('/home/sunyiyou/PycharmProjects/test/%03d.jpg' % 1)
        with open(settings.DATASET_INDEX_FILE) as f:
            image_list = f.readlines()
            predictions = []
            for i, file in enumerate(image_list):
                print("generating visualization on %03d" % i)
                image_file = os.path.join(settings.DATASET_PATH, file.strip())
                vis, prediction = fo.instance_cam_by_file(model, image_file, feat_clf)
                predictions.append(prediction)
                vis.save(os.path.join(settings.OUTPUT_FOLDER, 'html', 'image', "%03d.jpg" % i))

    elif settings.APP == "vqa":
        from loader.vqa_data_loader import VQA, collate_fn
        vqa_test = VQA(settings.VQA_QUESTIONS_FILE,settings.VQA_ANSWERS_FILE,settings.VQA_IMG_PATH)
        loader = torch.utils.data.DataLoader(
            vqa_test,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            collate_fn=collate_fn,
        )
        predictions = []
        for i, (v, q, a, idx, q_len, q_org, a_org) in enumerate(loader):
            print("generating visualization on %03d" % i)
            vis, prediction = fo.instance_cam_by_file(model, v[0], feat_clf, other_params=(q, q_len, a))
            vis.save(os.path.join(settings.OUTPUT_FOLDER, 'html', 'image', "%03d.jpg" % i))
            # question = []
            # for q_i in range(q_len[0]):
            #     question.append(vqa_test.questions_tokens[q[0][q_i] - 1])
            prediction_answer = vqa_test.answer_tokens[prediction.data[0]]
            predictions.append((' '.join([qw[0] for qw in q_org]), a_org[0][0], prediction_answer))
    # elif settings.APP == "imagecap":

    import pickle
    with open(os.path.join(settings.OUTPUT_FOLDER, 'prediction.pickle'), 'wb') as f:
        pickle.dump(predictions, f)

    return predictions


def fig_sample(fo, model, feat_clf):
    if settings.APP == "classification" or settings.APP == "imagecap":
        # image_file ='/home/sunyiyou/PycharmProjects/test/example4.jpg'
        # vis, prediction = fo.instance_cam_by_file(model, image_file, feat_clf, fig_style=1)
        # vis.save('test.jpg')
        with open(settings.DATASET_INDEX_FILE) as f:
            image_list = f.readlines()
            predictions = []
            outpath = os.path.join(settings.OUTPUT_FOLDER, 'html', 'image')
            if not os.path.exists(outpath):
                os.makedirs(outpath)
            for i, file in enumerate(image_list):
                print("generating figure on %03d" % i)
                image_file = os.path.join(settings.DATASET_PATH, file.strip())
                vis, prediction = fo.instance_cam_by_file(model, image_file, feat_clf, fig_style=1)
                predictions.append(prediction)
                vis.save(os.path.join(outpath, "%03d.jpg" % i))
    import pickle
    with open(os.path.join(settings.OUTPUT_FOLDER, 'prediction.pickle'), 'wb') as f:
        pickle.dump(predictions, f)

    return predictions
