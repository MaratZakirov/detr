import torchvision.transforms as T
import torchvision.transforms.functional as F
from torchvision.ops import nms
import matplotlib.pyplot as plt
from PIL import Image
import torch

# standard PyTorch mean-std input image normalization
transform = T.Compose([
    T.Resize((672, 672)),
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

    # propagate through the model
    outputs = model(img)

    # keep only predictions with 0.7+ confidence
    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > 0.4

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    return probas[keep], bboxes_scaled

def plot_results(pil_img, prob, boxes):
    plt.figure(figsize=(16,10))
    plt.imshow(pil_img)
    ax = plt.gca()
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes.tolist()):
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                   fill=False, color=(0, 0, 0), linewidth=3))
        cl = p.argmax()
        text = str(cl.item())
        ax.text(xmin, ymin, text, fontsize=15,
                bbox=dict(facecolor='yellow', alpha=0.5))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    iname = '/home/marat/281_MED_IMG_0158.JPG'
    #iname = '/home/marat/372_MED_DSC05594.JPG'
    #iname = '/home/marat/304_GOOD_DSC05502.jpg'
    model = torch.load('infer.model').to('cpu')

    image = Image.open(iname)

    #image = F.rotate(image, 20)

    scores, boxes = detect(image, model, transform)

    keep = nms(boxes, scores.max(dim=1).values, iou_threshold=0.3)
    scores = scores[keep]
    boxes = boxes[keep]

    plot_results(image, scores, boxes)
