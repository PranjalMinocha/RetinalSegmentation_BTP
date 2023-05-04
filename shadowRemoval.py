from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import numpy as np
import torch.utils.data
import cv2
import torchvision.models.segmentation
import torch
from tqdm import tqdm

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')   # train on the GPU or on the CPU, if a GPU is not available
device = torch.device('cpu')

def load_model():
  model = torchvision.models.detection.maskrcnn_resnet50_fpn(weights=None)  # load an instance segmentation model
  in_features = model.roi_heads.box_predictor.cls_score.in_features  # get number of input features for the classifier
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features,num_classes=2)  # replace the pre-trained head with a new one
  model.to(device)# move model to the right device

#   optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-5)
  model.load_state_dict(torch.load("Final app/shadowRCNN_2000.torch", map_location=device)) 
  model.eval() # set model to evaluation state

  return model

def preprocess(image):

  # Simar Changes
  # image = cv2.resize(image.transpose(1,2,0), imageSize[::-1],
  #               interpolation = cv2.INTER_LINEAR)
  if(len(image.shape) == 2):
    image = cv2.cvtColor(np.float32(image), cv2.COLOR_GRAY2BGR)
  image = torch.as_tensor(image, dtype=torch.float32).unsqueeze(0)
  image = image.swapaxes(1, 3).swapaxes(2, 3)
  # return list(image.to(device) for img in image)
  return image.to(device)


def remove(model, images):
    alpha = 2
    newImgs = []
    with torch.no_grad():
      for img in tqdm(images):
        pred = model(img)
        im = img[0].swapaxes(0, 2).swapaxes(0, 1).detach().cpu().numpy().astype(np.uint8)
          
        img_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        for idx in range(len(pred[0]['masks'])):
          msk=pred[0]['masks'][idx,0].detach().cpu().numpy()
          scr=pred[0]['scores'][idx].detach().cpu().numpy()
          if scr>0.5 :
            img_hsv[:, :, 2] += (alpha*msk*np.log2(256-img_hsv[:, :, 2])).astype('uint8')

        itr_img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)
        newImgs.append(itr_img)
    return newImgs