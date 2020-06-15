import torchvision.transforms as T
import torchvision.transforms.functional as F
from scipy.optimize import linear_sum_assignment
from torchvision.ops import nms, box_iou
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
import os

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize((800, 800)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# for output bounding box post-processing
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def detect(im, model, transform):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    if torch.cuda.is_available():
        img = img.cuda()

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.4

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep].cpu(), im.size)
    return probas[keep].cpu(), bboxes_scaled.cpu()

def CalcStat(pcls, pboxes, tcls, tboxes):
    tp_fn = len(tboxes)
    tp_fp = len(pboxes)

    dists = box_iou(pboxes, tboxes)
    assigns = linear_sum_assignment(-dists)

    tp = (pcls[assigns[0]] == tcls[assigns[1]]).sum().item()

    prescision = tp / (tp_fp + 0.001)
    recall = tp / (tp_fn + 0.001)
    f1_score = 2 * recall * prescision / (recall + prescision + 0.001)

    return f1_score, prescision, recall, assigns

def plot_results(iname, pil_img, prob, boxes, tboxes, tclasses):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()

    # TODO: for now just throw away -1 class (ignore)
    tboxes = tboxes[tclasses >= 0]
    tclasses = tclasses[tclasses >= 0]

    f1_score, prescision, recall, assigns = CalcStat(prob.argmax(1), boxes, tclasses, tboxes)

    # printing TP
    for i, j in zip(assigns[0], assigns[1]):
        xmin, ymin, xmax, ymax = boxes[i]
        p_cls = prob[i].argmax().item()
        t_cls = int(tclasses[j].item())
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False,
                                   color=(0, 1, 0) if p_cls == t_cls else (1, 0, 0), linewidth=1))
        ax.text(xmin, ymin, str(p_cls), fontsize=15,
                bbox=dict(facecolor='green' if p_cls == t_cls else 'red', alpha=0.5))

    # Printing FN
    for i in range(len(tboxes)):
        if i not in assigns[1]:
            xmin, ymin, xmax, ymax = tboxes[i]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=(1, 1, 1), linewidth=1))
            ax.text(xmax, ymax, str(int(tclasses[i].item())), fontsize=15, bbox=dict(facecolor='white', alpha=0.5))

    # Printing FP
    for i in range(len(boxes)):
        if i not in assigns[0]:
            xmin, ymin, xmax, ymax = boxes[i]
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=(1, 1, 1), linewidth=1))
            ax.text(xmax, ymax, prob[i].argmax().item(), fontsize=15, bbox=dict(facecolor='red', alpha=0.5))

    plt.axis('off')
    plt.savefig('outputs/' + iname.split('/')[-1])
    #plt.show()

    return f1_score, prescision, recall

def openDataset(inames):
    dataset = []
    for image_name in open(inames, 'r').readlines():
        label_name = image_name.replace('images', 'labels').replace('jpg','txt')
        targets = torch.tensor(np.loadtxt(label_name.rstrip()).astype(np.float32))
        dataset.append((image_name.rstrip(), targets))
    return dataset

if __name__ == '__main__':
    # TODO here is the list of worst images for yolo3 detector
    """
    inames = [
        '/home/marat/281_MED_IMG_0158.JPG',
        '/home/marat/372_MED_DSC05594.JPG',
        '/home/marat/304_GOOD_DSC05502.jpg',
        "/home/marat/dataset/photo_birka/part1/images/1038_GOOD_IMG_3574.jpg",
        "/home/marat/dataset/photo_birka/part1/images/694_GOOD_IMG_3510.jpg",
        "/home/marat/dataset/photo_birka/part1/images/272_GOOD_IMG_0244.jpg",
        "/home/marat/dataset/photo_birka/part1/images/550_GOOD_IMG_3159.jpg",
        "/home/marat/dataset/photo_birka/part1/images/1033_GOOD_IMG_3562.jpg",
        "/home/marat/dataset/photo_birka/part1/images/43_GOOD_IMG_0211.jpg",
        "/home/marat/dataset/photo_birka/part1/images/250_GOOD_IMG_0092.jpg",
        "/home/marat/dataset/photo_birka/part1/images/44_GOOD_IMG_0212.jpg",
        "/home/marat/dataset/photo_birka/part1/images/632_GOOD_IMG_3663.jpg",
        "/home/marat/dataset/photo_birka/part1/images/1032_GOOD_IMG_3559.jpg",
        "/home/marat/dataset/photo_birka/part1/images/389_GOOD_IMG_0071.jpg",
        "/home/marat/dataset/photo_birka/part1/images/551_GOOD_IMG_3163.jpg",
        "/home/marat/dataset/photo_birka/part1/images/503_GOOD_IMG_3035.jpg"
    ]
    """

    inames = '/home/marat/dataset/photo_birka/part1/test.txt.true'
    dataset = openDataset(inames)#[:2]

    os.makedirs('outputs', exist_ok=True)

    model = torch.load('inferYOLO3.model')

    if torch.cuda.is_available():
        model = model.cuda()

    f1pr = []

    for i, (iname, targets) in enumerate(dataset):
        print('processing:', iname, i, '/', len(dataset))
        image = Image.open(iname)
        #image = F.rotate(image, 20)

        tboxes = targets[:, 3:]
        tboxes = torch.cat([tboxes[:, :2] - tboxes[:, 2:] / 2, tboxes[:, :2] + tboxes[:, 2:] / 2], 1)
        tboxes[:, 0::2] *= image.width
        tboxes[:, 1::2] *= image.height

        tclasses = targets[:, 2]

        with torch.no_grad():
            scores, boxes = detect(image, model, transform)

            keep = nms(boxes, scores.max(dim=1).values, iou_threshold=0.3)
            scores = scores[keep]
            boxes = boxes[keep]

            f1pr.append(plot_results(iname, image, scores, boxes, tboxes, tclasses))

    f1pr = np.array(f1pr)
    for i in np.argsort(f1pr[:, 0]):
        iname = dataset[i][0]
        print('name: ', iname, 'f1:', f1pr[i][0], ' p: ', f1pr[i][1], ' r: ', f1pr[i][2])

    print('===== F1:', f1pr[:, 0].mean(), ' prescision: ', f1pr[:, 1].mean(), ' recall: ', f1pr[:, 2].mean())

"""
Results for resnet50_32
===== F1: 0.8937709339859103  prescision:  0.901109333618579  recall:  0.8925354267822092
"""

"""
Results for resnet50_16
===== F1: 0.8973964776760863  prescision:  0.9012736928754705  recall:  0.8993737186780283
"""

"""
Results for yolo3
===== F1: 0.29414858733629645  prescision:  0.5090641240183771  recall:  0.21356210417920213
"""
