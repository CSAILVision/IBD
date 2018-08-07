######### global settings  #########
GPU = True                                                                   # running on GPU is highly suggested
CLEAN = False                                                                # set to "True" if you want to clean the temporary large files after generating result
APP = "classification"                                                       # Do not change! mode choide: "classification", "imagecap", "vqa". Currently "imagecap" and "vqa" are not supported.
CATAGORIES = ["object", "part"]                                              # Do not change! concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"

CAM_THRESHOLD = 0.5                                                          # the threshold used for CAM visualization
FONT_PATH = "components/font.ttc"                                            # font file path
FONT_SIZE = 26                                                               # font size
SEG_RESOLUTION = 7                                                           # the resolution of cam map
BASIS_NUM = 7                                                                # In decomposition, this is to decide how many concepts are used to interpret the weight vector of a class.

EPOCHS = 15                                                                  # max epochs to train the concept classifier
SNAPSHOT_FREQ = 5                                                            # the frequence of making snapshot
SINGLE_LABEL = False                                                         # Do not change.
COMPRESSED_INDEX = True                                                      # Do not change.
WORKERS = 4                                                                  # how much thread is used to extract images
BATCH_SIZE = 128                                                             # batch size when extracting image feature
FEAT_BATCH_SIZE = 16                                                         # feature's batch in training feature classifier
TALLY_BATCH_SIZE = 4                                                         # batch size when tallying concept
TALLY_AHEAD = 4                                                              # size of prefetching batch when tallying concept
INDEX_FILE = 'index.csv'                                                     # image index of concept dataset

CAFFE_MODEL = False                                                          # whether the model is transferred from "*.caffemodel".
CNN_MODEL = 'resnet18'                                                       # model arch: resnet18, alexnet, resnet50, densenet161, etc...
DATASET = 'places365'                                                        # model trained on: places365 or imagenet
OUTPUT_FOLDER = "result/pytorch_"+CNN_MODEL+"_"+DATASET                      # where output file exists

DATASET_PATH = 'components/sample_images'                                    # where sample image folder exists
DATASET_INDEX_FILE = 'components/sample_images/sample.txt'                   # a file list of sample images
if (not CNN_MODEL.endswith('CAM')) and (CNN_MODEL == "alexnet" or CNN_MODEL.startswith('vgg')):
    GRAD_CAM = True                                                          # to decide if we have to decompose the grad-CAM for the chosen model
else:
    GRAD_CAM = False
if DATASET == 'places365':
    NUM_CLASSES = 365                                                        # class amount of dataset
    if CNN_MODEL == 'resnet18':
        MODEL_FILE = 'zoo/resnet18_places365.pth.tar'                        # model filee's path
        MODEL_PARALLEL = True                                                # if the model is trained by multi-GPUs


if CNN_MODEL != 'alexnet' and CNN_MODEL != 'caffenetCAM':
    DATA_DIRECTORY = 'dataset/broden1_224'                                   # concept dataset's path
    IMG_SIZE = 224                                                           # image's size in the concept dataset
else:
    DATA_DIRECTORY = 'dataset/broden1_227'
    IMG_SIZE = 227

if CAFFE_MODEL:
    FEATURE_NAMES = ['pool5']                                                # the layer to be decomposed
else:
    if 'resnet' in CNN_MODEL:
        FEATURE_NAMES = ['layer4']
    elif CNN_MODEL == 'densenet161' or CNN_MODEL == 'alexnet' or CNN_MODEL.startswith('vgg'):
        FEATURE_NAMES = ['features']


