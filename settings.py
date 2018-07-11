######### global settings  #########
GPU = True     # running on GPU is highly suggested
TEST_MODE = False                            # turning on the testmode means the code will run on a small dataset.
CLEAN = False                               # set to "True" if you want to clean the temporary large files after generating result
CATAGORIES = ["object", "part"]             # concept categories that are chosen to detect: "object", "part", "scene", "material", "texture", "color"
APP = "classification"                      # "classification", "imagecap", "vqa"

QUANTILE = 0.005                            # the threshold used for activation
CAM_THRESHOLD = 0.5
SEG_THRESHOLD = 0.04                        # the threshold used for visualization
CONCEPT_TOPN = 5
FONT_PATH = "components/font.ttc"
FONT_SIZE = 26

EPOCHS = 15
SNAPSHOT_FREQ = 5
SINGLE_LABEL = False
SEG_RESOLUTION = 7
COMPRESSED_INDEX = True
BASIS_NUM = 7


if APP == "classification":
    CAFFE_MODEL = False
    CNN_MODEL = 'resnet18'  # model arch: wide_resnet18, resnet18, alexnet, resnet50, densenet161, vgg16CAM, caffenetCAM
    DATASET = 'places365'  # model trained on: places365 or imagenet
    OUTPUT_FOLDER = "result/pytorch_"+CNN_MODEL+"_"+DATASET

    DATASET_PATH = 'components/sample_images'
    DATASET_INDEX_FILE = 'components/sample_images/sample.txt'
    if (not CNN_MODEL.endswith('CAM')) and (CNN_MODEL == "alexnet" or CNN_MODEL.startswith('vgg')):
        GRAD_CAM = True
    else:
        GRAD_CAM = False
    if DATASET == 'places365':
        NUM_CLASSES = 365
        if CNN_MODEL == 'resnet18':
            MODEL_FILE = 'zoo/resnet18_places365.pth.tar'
            MODEL_PARALLEL = True
        elif CNN_MODEL == 'wideresnet18':
            MODEL_FILE = 'zoo/whole_wideresnet18_places365_python36.pth.tar'
            MODEL_PARALLEL = True
        else:
            MODEL_FILE = 'zoo/whole_'+CNN_MODEL+'_places365_python36.pth.tar'
            MODEL_PARALLEL = False
    elif DATASET == 'imagenet':
        NUM_CLASSES = 1000
        MODEL_FILE = None
        MODEL_PARALLEL = False
    if CAFFE_MODEL:
        if CNN_MODEL == "caffenetCAM":
            # PROTOTXT_PATH = 'zoo/deploy_caffeNetCAM_places365.prototxt'
            MODEL_FILE = 'zoo/caffenetCAM.pth'
        elif CNN_MODEL == "vgg16CAM":
            # PROTOTXT_PATH = 'zoo/deploy_vgg16CAM_places365.prototxt'
            MODEL_FILE = 'zoo/vgg16CAM.pth'
        elif CNN_MODEL == "alexnet":
            MODEL_FILE = 'zoo/caffenet_places365.pth'
        elif CNN_MODEL == "vgg16":
            MODEL_FILE = 'zoo/vgg16_places365.pth'

if APP == "imagecap":
    CAM = False
    GRAD_CAM = True
    CNN_MODEL = 'resnet18'
    CAFFE_MODEL = False
    MODEL_FILE = 'zoo/imagecap.pth.tar'
    OUTPUT_FOLDER = "result/imagecap_"+CNN_MODEL
    MAX_SENT_LEN = 20
    DATASET_PATH = '/home/sunyiyou/dataset/places365_standard'
    DATASET_INDEX_FILE = '/home/sunyiyou/dataset/places365_standard/val_sample.txt'

if APP == "vqa":
    CAM = False
    GRAD_CAM = True
    CNN_MODEL = 'resnet152'
    CAFFE_MODEL = False
    DATASET = 'imagenet'
    MODEL_FILE = 'zoo/vqa_20.pth'
    OUTPUT_FOLDER = "result/vqa_"+CNN_MODEL
    IMG_SIZE = 224
    DATA_DIRECTORY = '../NetDissect-Lite/dataset/broden1_224'
    OUTPUT_FEATURE_SIZE = 2048
    MAX_ANSWERS = 3000
    VQA_IMG_PATH = '/home/sunyiyou/dataset/coco/test_vqa/'
    VQA_QUESTIONS_FILE = '/home/sunyiyou/dataset/coco/test_vqa/sample_OpenEnded_questions.json'
    VQA_ANSWERS_FILE = '/home/sunyiyou/dataset/coco/test_vqa/sample_answers.json'
    VQA_IMAGE_INDEX_FILE = '/home/sunyiyou/dataset/coco/test_vqa/sample_coco_img.npy'
    VOCAB_FILE = '/home/sunyiyou/PycharmProjects/NetDissect2/components/vocab.json'


if APP != "vqa":
    if CNN_MODEL != 'alexnet' and CNN_MODEL != 'caffenetCAM':
        DATA_DIRECTORY = 'dataset/broden1_224'
        IMG_SIZE = 224
    else:
        DATA_DIRECTORY = 'dataset/broden1_227'
        IMG_SIZE = 227

if CAFFE_MODEL:
    if CNN_MODEL.endswith('CAM'):
        FEATURE_NAMES = ['CAM_conv']
    else:
        FEATURE_NAMES = ['pool5']

else:
    if 'resnet' in CNN_MODEL:
        FEATURE_NAMES = ['layer4']
    elif CNN_MODEL == 'densenet161' or CNN_MODEL == 'alexnet' or CNN_MODEL.startswith('vgg'):
        FEATURE_NAMES = ['features']



if TEST_MODE:
    WORKERS = 1
    BATCH_SIZE = 4
    FEAT_BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 2
    TALLY_AHEAD = 1
    INDEX_FILE = 'index_sm.csv'
    OUTPUT_FOLDER += "_test"
else:
    WORKERS = 4
    BATCH_SIZE = 128
    FEAT_BATCH_SIZE = 16
    TALLY_BATCH_SIZE = 16
    TALLY_AHEAD = 4
    INDEX_FILE = 'index.csv'
