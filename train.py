import os
import settings
from loader.model_loader import loadmodel
from util.feature_operation import FeatureOperator
from util.feature_decoder import SingleSigmoidFeatureClassifier
from util.image_operation import *

model = loadmodel()
fo = FeatureOperator()
features, _ = fo.feature_extraction(model=model)

for layer_id, layer in enumerate(settings.FEATURE_NAMES):
    settings.GPU = False
    feat_clf = SingleSigmoidFeatureClassifier(feature=features[layer_id], layer=layer, fo=fo)
    feat_clf.run()


