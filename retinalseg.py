import numpy as np
import cv2
from tensorflow import keras
import os
from datetime import datetime

os.environ["SM_FRAMEWORK"] = "tf.keras"

import segmentation_models as sm
from sklearn.model_selection import train_test_split

# classes for data loading and preprocessing
class TrainDataset:
    """
    Args:
        img_stk (tiff)
        mask_stk (tiff)
    """
    
    def __init__(self, img_stk, mask_stk):
        self.imgs = img_stk
        self.masks = mask_stk
    
    def __getitem__(self, i):
        
        # read data
        image = self.imgs[i]
        image = cv2.resize(image, (512,512))
        if(len(image.shape) < 3):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        mask = self.masks[i]
        mask = cv2.resize(mask, (512,512))

        return image.astype(np.float32), mask.astype(np.float32)
        
    def __len__(self):
        return len(self.imgs)

class TestDataset:
    """
    Args:
        img_stk (tiff)
    """
    
    def __init__(self, img_stk):
        self.imgs = img_stk
    
    def __getitem__(self, i):
        
        # read data
        image = self.imgs[i]
        image = cv2.resize(image, (512,512))
        if(len(image.shape) < 3):
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        return image.astype(np.float32)
        
    def __len__(self):
        return len(self.imgs)
    
    
class Dataloder(keras.utils.Sequence):
    """Load data from dataset and form batches
    
    Args:
        dataset: instance of Dataset class for image loading and preprocessing.
        batch_size: Integet number of images in batch.
        shuffle: Boolean, if `True` shuffle image indexes each epoch.
    """
    
    def __init__(self, dataset, batch_size=1, shuffle=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(dataset))

        self.on_epoch_end()

    def __getitem__(self, i):
        
        # collect batch data
        start = i * self.batch_size
        stop = (i + 1) * self.batch_size
        data = []
        for j in range(start, stop):
            data.append(self.dataset[j])
        
        # transpose list of lists
        batch = [np.stack(samples, axis=0) for samples in zip(*data)]
        
        return batch
    
    def __len__(self):
        """Denotes the number of batches per epoch"""
        return len(self.indexes) // self.batch_size
    
    def on_epoch_end(self):
        """Callback function to shuffle indexes each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(self.indexes)

# helper function for data denormalization    
def denormalize(x):
    """Scaling image to range 0-1 for correct plot"""
    x_max = np.percentile(x, 98)
    x_min = np.percentile(x, 2)    
    x = (x - x_min) / (x_max - x_min)
    x = x.clip(0, 1)
    return x

# helper function to get a model object 
def get_model(path):
    BACKBONE = 'efficientnetb3'
    LR = 0.00001 #decreased the LR by 10x

    # defining network parameters
    n_classes = 1 
    activation = 'sigmoid'

    # creating model
    model = sm.Unet(BACKBONE, classes=n_classes, activation=activation)

    # defining optimizer
    optim = keras.optimizers.legacy.Adam(LR)

    # Segmentation models losses can be combined together by '+'
    total_loss = sm.losses.DiceLoss() + sm.losses.BinaryFocalLoss()
    metrics = [sm.metrics.IOUScore(), sm.metrics.FScore()]

    # compiling keras model with defined optimozer, loss and metrics
    model.compile(optim, total_loss, metrics)

    if(path != None):
        model.load_weights("Final app/saved_models/"+path)
    return model

#Custom callback
class CustomSaver(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(epoch%5 == 4):
            now = datetime.now()
            time_str = now.strftime("%m%d%Y_%Hh%Mm%Ss")
            self.model.save("Final app/saved_models/"+time_str+"_segModel_{}.h5".format(epoch+1))

def train(imgs, masks, model, epochs):
    # In the first step we will split the data in training and validation dataset
    X_train, X_val, y_train, y_val = train_test_split(imgs, masks, train_size=0.8)

    # Dataset for train images
    train_dataset = TrainDataset(X_train, y_train)
    val_dataset = TrainDataset(X_val, y_val)


    train_dataloader = Dataloder(train_dataset, batch_size=2, shuffle=True)
    valid_dataloader = Dataloder(val_dataset, batch_size=1, shuffle=False)

    saver = CustomSaver()

    # train model
    history = model.fit(
        train_dataloader, 
        steps_per_epoch=len(train_dataloader), 
        epochs=epochs, 
        callbacks=[saver],
        validation_data=valid_dataloader, 
        validation_steps=len(valid_dataloader))

    return (history.history['iou_score'], history.history['val_iou_score']), (history.history['loss'], history.history['val_loss']), model.predict(train_dataloader)

def test(img_stk, model):
    test_dataset = TestDataset(img_stk)
    pred = model.predict(np.asarray(test_dataset))
    return pred