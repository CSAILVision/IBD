import PIL
from PIL import ImageDraw, ImageFont
import skimage.measure
import numpy as np
import settings
import cv2
from scipy.misc import imresize
from PIL import Image

def imconcat(imgs, w, h, margin=0):
    w = sum([img.width for img in imgs])
    ret = PIL.Image.new("RGB", (w + (len(imgs) - 1) * margin, imgs[0].height), color=(255,255,255))
    w_pre = 0
    for i, img in enumerate(imgs):
        ret.paste(img, (w_pre+margin*int(bool(i)), 0))
        w_pre += img.width+margin*int(bool(i))
    # ret = PIL.Image.new("RGB", (len(imgs) * w + (len(imgs) - 1) * margin,h), color=(255,255,255))
    # for i, img in enumerate(imgs):
    #     ret.paste(img, ((w+margin)*i,0))
    return ret


def imstack(imgs):
    h = sum([img.height for img in imgs])
    ret = PIL.Image.new("RGB", (imgs[0].width, h))
    h_pre = 0
    for i, img in enumerate(imgs):
        ret.paste(img, (0, h_pre))
        h_pre += img.height
    return ret



def vis_cam_mask(cam_mat, org_img, vis_size, font_text=None):
    cam_mask = 255 * imresize(cam_mat, (settings.IMG_SIZE, settings.IMG_SIZE), mode="F")
    cam_mask = cv2.applyColorMap(np.uint8(cam_mask), cv2.COLORMAP_JET)[:, :, ::-1]
    vis_cam = cam_mask * 0.5 + org_img * 0.5
    vis_cam = Image.fromarray(vis_cam.astype(np.uint8))
    vis_cam = vis_cam.resize((vis_size, vis_size), resample=Image.BILINEAR)

    # if font_text is not None:
    #     font = ImageFont.truetype(settings.FONT_PATH, settings.FONT_SIZE+4)
    #     draw = ImageDraw.Draw(vis_cam)
    #     fw, fh = draw.textsize(font_text)
    #     coord = np.array(np.unravel_index(cam_mat.argmax(),cam_mat.shape)) * vis_size / cam_mat.shape[0]
    #     draw.text((coord[1], coord[0]), font_text, font=font, fill=(240, 240, 240, 255))

    return vis_cam

def label_seg(img, vis_size, labels, concept_inds, cam=None):
    h, w = concept_inds.shape
    grid_size = vis_size / settings.SEG_RESOLUTION
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(settings.FONT_PATH, settings.FONT_SIZE)
    X, Y = np.meshgrid(np.arange(7) * grid_size, np.arange(7) * grid_size)

    label_img = skimage.measure.label(concept_inds, connectivity=1)
    cpt_groups = skimage.measure.regionprops(label_img)
    for cpt_group in cpt_groups:
        # y_start, x_start, y_end, x_end = cpt_group.bbox
        label = labels[concept_inds[tuple(cpt_group.coords[0])]]['name']
        fw, fh = draw.textsize(label)
        coord = np.array(cpt_group.centroid)[::-1] * grid_size
        draw.text((coord[0] + (grid_size - fw) / 2, coord[1] + (grid_size - fh) / 2 ), label, font=font, fill=(0,0,0, 255))

    contours = skimage.measure.find_contours(cam, cam.max() * settings.CAM_THRESHOLD)
    for contour in contours:
        draw.line(list((contour[:,::-1] * vis_size / cam.shape[0]).ravel()), fill=(255, 200, 0))


def headline(captions, vis_size, height, width, margin=3):
    vis_headline = Image.fromarray(np.full((height, width, 3), 255, dtype=np.int8), mode="RGB")
    draw = ImageDraw.Draw(vis_headline)
    font = ImageFont.truetype(settings.FONT_PATH, settings.FONT_SIZE)
    for i in range(len(captions)):
        label = captions[i]
        fw, fh = draw.textsize(label)
        coord = ((vis_size+margin) * i, 0)
        draw.text((coord[0] + (vis_size - fw) / 2, coord[1] + (height - fh) / 2), label, font=font,
                  fill=(0, 0, 0, 255))
    return vis_headline

def headline2(captions, vis_size, height, width, margin=3):
    vis_headline = Image.fromarray(np.full((height, width, 3), 255, dtype=np.int8), mode="RGB")
    draw = ImageDraw.Draw(vis_headline)
    font = ImageFont.truetype(settings.FONT_PATH, settings.FONT_SIZE)
    for i in range(len(captions)):
        label = captions[i]
        fw, fh = draw.textsize(label)
        if i == 0:
            draw.text(((vis_size * 2 - fw*1.85) / 2, (height - fh*1.85) / 2), label, font=font, fill=(0, 0, 0, 255))
        else:
            coord = (vis_size *7 // 3 + (vis_size + margin) * (i - 1), 0)
            draw.text((coord[0] + (vis_size - fw*1.85) / 2, coord[1] + (height - fh*1.85) / 2), label, font=font, fill=(0, 0, 0, 255))

    return vis_headline

def big_margin(vis_size):
    w = vis_size // 3
    h = vis_size
    canvas = Image.fromarray(np.full((h, w, 3), 255, dtype=np.int8), mode="RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype(settings.FONT_PATH, settings.FONT_SIZE)
    label = "="
    draw.text(((w-settings.FONT_SIZE*0.5) / 2, (vis_size - settings.FONT_SIZE * 1.8) / 2), label, font=font, fill=(0, 0, 0, 255))
    return canvas