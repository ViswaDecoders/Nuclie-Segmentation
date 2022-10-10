import os
import cv2 
import numpy as np
from skimage.util.shape import view_as_windows
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from keras.models import load_model
import matplotlib.pyplot as plt
import segmentation_models as sm
import staintools

ROOT_DIR = "/content/drive/MyDrive/NucleiSegmentation/"

import sys
sys.path.append(os.path.join(ROOT_DIR, "src/utils/")) 
from metricfunctions import dice_coef,f1


class pipeline:
    
    def __init__(self):
        
        self.target = os.path.join(ROOT_DIR, "data","raw","MOSID","Training","TissueImages","TCGA-B0-5711-01Z-00-DX1.png")
        self.normalizer = staintools.StainNormalizer(method='vahadane')
        target_image = staintools.read_image(self.target)
        target_image = staintools.LuminosityStandardizer.standardize(target_image)
        self.normalizer.fit(target_image)
        
        custom_objs = {
            'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 
            'iou_score' : sm.metrics.iou_score,
            'dice_coef' : dice_coef,
            'f1' : f1
        }
        self.backbones = ["resnet101", "inceptionresnetv2", "densenet121"]
        self.models = [load_model(os.path.join(ROOT_DIR,"models","weights","MOSID-UNet-" + backbone + "-best.h5"),custom_objects= custom_objs, compile = False) for backbone in self.backbones]
        self.weights = [0.3, 0.3, 0.1]

    
    def create_patches(self, image, patch_width=256, patch_height=256):
        size = image.shape[0]
        sizeNew = 0
        for x in range(0, 100):
            if size < 2 ** x:
                sizeNew = 2 ** x
                break

        pad = sizeNew - size
        imgNew = np.zeros((sizeNew, sizeNew,3))
        imgNew[pad // 2 : sizeNew - (pad // 2), pad // 2 : sizeNew - (pad // 2), :] = image

        new_imgs = view_as_windows(imgNew, (patch_width, patch_height, 3), (patch_width, patch_height, 3))
        new_imgs = new_imgs.reshape(-1, patch_width, patch_height, 3)

        return new_imgs

    def merge_patches(self, patches, image_width, patch_width=256, patch_height=256):
        num = int(np.sqrt(patches.shape[0]))
        img = np.zeros((patch_width * num, patch_height * num, 1))
        count = 0
        for x in range(0, num):
            for y in range(0, num):
                startX = x * patch_width
                endX = (x + 1)* patch_width
                startY = y * patch_width
                endY = (y + 1) * patch_width

                img[startX : endX, startY : endY, :]= patches[count]
                count = count + 1
        
        pad = img.shape[0] - image_width
        img = img[pad // 2 : image_width + (pad // 2), pad // 2 : image_width + (pad // 2), :]
        return img
    
    def preprocess(self, input_file):

        image = staintools.read_image(input_file)
        image_width = np.max([image.shape[0],image.shape[1]])
        image = cv2.resize(image,(image_width,image_width))
        image = staintools.LuminosityStandardizer.standardize(image)
        normalized_image = self.normalizer.transform(image)
        
        return self.create_patches(normalized_image), image_width
    
    def ensemble_augmented_preds(self, pred_masks):
        n = len(pred_masks)
        fusion = np.sum(pred_masks, axis=0)
        return np.where(fusion >= n / 2, 1, 0)

    def augment(self, img):
        augmentations = []
        augmentations.append(img)
        augmentations.append(cv2.flip(img, 0))
        augmentations.append(cv2.flip(img, 1))
        augmentations.append(cv2.flip(img, -1))
        augmentations.append(cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE))
        augmentations.append(cv2.rotate(img, cv2.ROTATE_180))
        augmentations.append(cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE))
        return np.array(augmentations)

    def undo_augmentations(self, imgs):
        origs = []
        origs.append(imgs[0])
        origs.append(cv2.flip(imgs[1],0))
        origs.append(cv2.flip(imgs[2],1))
        origs.append(cv2.flip(imgs[3],-1))
        origs.append(cv2.rotate(imgs[4], cv2.cv2.ROTATE_90_COUNTERCLOCKWISE))
        origs.append(cv2.rotate(imgs[5], cv2.cv2.ROTATE_180))
        origs.append(cv2.rotate(imgs[6], cv2.cv2.ROTATE_90_CLOCKWISE))
        for i in range(1,len(origs)): origs[i] = np.expand_dims(origs[i], axis=2)
        return np.array(origs)
    
    def predict(self, patches):

        processed_patches = []
        for backbone in self.backbones:
            preprocess_input =  sm.get_preprocessing(backbone)
            processed_patches.append(preprocess_input(patches))
        
        y_pred = []
        for i in range(len(processed_patches)):
            model_inputs = [self.augment(processed_patches[k][i]) for k in range(len(self.models))]
            aug_preds = []
            for j in range(len(model_inputs[0])):
                individual_model_preds = np.array([self.models[k].predict(np.expand_dims(model_inputs[k][j], axis=0)) for k in range(len(self.models))])
                weighted_preds = np.round(np.tensordot(individual_model_preds, self.weights, axes=((0),(0))))
                aug_preds.append(weighted_preds[0])
            y_pred.append(self.ensemble_augmented_preds(self.undo_augmentations(aug_preds)))

        y_pred = np.array(y_pred).astype('float32')

        # final_pred = self.merge_patches(y_pred, self.image_width)
        return final_pred
    
    def run(self, input_file):
        
        patches, image_width = self.preprocess(input_file)
        self.image_width = image_width
        self.final_pred = self.predict(patches)

        # fig, axs = plt.subplots(1,2)
        # axs[0].imshow(load_img(input_file))
        # axs[1].imshow(array_to_img(final_pred))






        
