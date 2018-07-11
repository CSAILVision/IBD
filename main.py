
import settings
import os
from loader.model_loader import loadmodel
from util.feature_operation import FeatureOperator
from util.feature_decoder import SingleSigmoidFeatureClassifier
from visualize.plot import fig_sample
from visualize.html import html_cam_seg

model = loadmodel()
fo = FeatureOperator()

features, _ = fo.feature_extraction(model=model)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    feat_clf = SingleSigmoidFeatureClassifier(feature=features[layer_id], layer=layer, fo=fo)
    feat_clf.load_snapshot(14, unbiased=True)

    # feat_clf.run()
    if not settings.GRAD_CAM:
        fo.weight_decompose(model, feat_clf, feat_labels=[l['name'] for l in fo.data.label])
    predictions = fig_sample(fo, model, feat_clf)
    html_cam_seg(os.path.join(settings.OUTPUT_FOLDER, 'html', 'result.html'), 'image', range(len(predictions)), predictions)


