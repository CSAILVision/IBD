import os
import settings
def html_cam_seg(output_file, image_folder, inds, info):
    f = open(output_file, 'w')
    f.write("<!DOCTYPE html>\n<html lang='en'>\n<head>\n<meta charset='UTF-8'>\n<title>\n</title>\n</head>\n<body>\n")
    # f.write("<img height='227' src='image/accuracy_distribute.jpg'>")
    for ind in inds:
        if settings.APP == "vqa":
            headline = "<h3>%d  %s? real: %s, prediction %s </h2><br>\n" % (ind, info[ind][0], info[ind][1], info[ind][2])
        elif settings.APP == "imagecap":
            sents, highlight_id = info[ind]
            sents = list(sents)
            sents[highlight_id] = "<font color='red'>%s</font>" % sents[highlight_id]
            headline = "<h3>%d  %s</h2><br>\n" % (ind, ' '.join(sents))
        else:
            headline = "<h3>%d  %s</h2><br>\n" % (ind, info[ind])
        imageline = "<img id='%d' height='%d' src='%s'>\n" % (ind, settings.IMG_SIZE, os.path.join(image_folder, '%03d.jpg' % ind))
        f.write(headline)
        f.write(imageline)
    f.write("</body>\n</html>\n")
    f.close()
