import settings
import torch
import torchvision
import torch.nn as nn
from util.feature_operation import hook_feature, hook_grad

def loadmodel():
    if settings.APP == "vqa":
        from loader.vqa_resnet import resnet152
        from loader.vqa_model import VQANet, SimpleVQANet
        net_status = torch.load(settings.MODEL_FILE)
        vqa_net = nn.DataParallel(SimpleVQANet(net_status['metadata']['num_tokens'])).cuda()
        vqa_net.load_state_dict(net_status['weights'])
        vqa_net.cnn = resnet152(pretrained=True)
        for name in settings.FEATURE_NAMES:
            vqa_net.cnn._modules.get(name).register_forward_hook(hook_feature)
            vqa_net.cnn._modules.get(name).register_backward_hook(hook_grad)
        # if settings.GPU:
        #     vqa_net.cnn.cuda()
        model = vqa_net
    elif settings.APP == "imagecap":
        from loader.caption_model import CaptionModel
        model = CaptionModel()
        model.load_checkpoint(settings.MODEL_FILE)
        for name in settings.FEATURE_NAMES:
            model.cnn._modules.get(name).register_forward_hook(hook_feature)
            model.cnn._modules.get(name).register_backward_hook(hook_grad)
    elif settings.APP == "classification":
        if settings.CAFFE_MODEL:
            from loader.caffe_model import CaffeNet_David, VGG16, CaffeNetCAM, VGG16CAM
            if settings.CNN_MODEL == "alexnet":
                model = CaffeNet_David()
            elif settings.CNN_MODEL == "vgg16":
                model = VGG16()
            elif settings.CNN_MODEL == "caffenetCAM":
                model = CaffeNetCAM()
            elif settings.CNN_MODEL == "vgg16CAM":
                model = VGG16CAM()
            model.load_state_dict(torch.load(settings.MODEL_FILE))
        else:
            if settings.MODEL_FILE is None:
                model = torchvision.models.__dict__[settings.CNN_MODEL](pretrained=True)
            else:
                checkpoint = torch.load(settings.MODEL_FILE)
                if type(checkpoint).__name__ == 'OrderedDict' or type(checkpoint).__name__ == 'dict':
                    model = torchvision.models.__dict__[settings.CNN_MODEL](num_classes=settings.NUM_CLASSES)
                    if settings.MODEL_PARALLEL:
                        state_dict = {str.replace(k, 'module.', ''): v for k, v in checkpoint[
                            'state_dict'].items()}  # the data parallel layer will add 'module' before each layer name
                    else:
                        state_dict = checkpoint
                    model.load_state_dict(state_dict)
                else:
                    model = checkpoint
        for name in settings.FEATURE_NAMES:
            model._modules.get(name).register_forward_hook(hook_feature)
            model._modules.get(name).register_backward_hook(hook_grad)
    if settings.GPU:
        model.cuda()
    model.eval()
    return model
