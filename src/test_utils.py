import os
from turtle import undo
import cv2
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm.notebook import tqdm
from keras.models import load_model
from evaluation_metrics import *

ROOT_DIR = '/content/drive/My Drive/NucleiSegmentation'
os.chdir(ROOT_DIR)

import segmentation_models as sm

custom_objs = {
    'binary_crossentropy_plus_jaccard_loss': sm.losses.bce_jaccard_loss, 
    'iou_score' : sm.metrics.iou_score,
    'dice_coef' : dice_coef,
    'f1' : f1
}

def ensemble_augmented_preds(pred_masks):
  n = len(pred_masks)
  fusion = np.sum(pred_masks, axis=0)
  return np.where(fusion >= n / 2, 1, 0)

def augment_image(img):
  augmentations = []
  augmentations.append(img)
  augmentations.append(cv2.flip(img, 0))
  augmentations.append(cv2.flip(img, 1))
  augmentations.append(cv2.flip(img, -1))
  augmentations.append(cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE))
  augmentations.append(cv2.rotate(img, cv2.ROTATE_180))
  augmentations.append(cv2.rotate(img, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE))
  return np.array(augmentations)

def undo_augmentations(imgs):
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

def get_metrics(y_test, y_pred):
  metrics = {}
  y_pred_round = np.round(y_pred)
  metrics['accuracy'] = accuracy(y_test, y_pred_round)
  metrics['precision'] = precision(y_test, y_pred_round).numpy()
  metrics['recall'] = recall(y_test, y_pred_round).numpy()
  metrics['f1'] = f1(y_test, y_pred_round).numpy()
  metrics['dice'] = dice_coef(y_test, y_pred_round).numpy()
  metrics['iou'] = IoU_coef(y_test, y_pred_round).numpy()
  
  return metrics

def get_ensemble_weights(y_test, preds):
  
  df = pd.DataFrame([])
  for w1 in range(0, 4):
      for w2 in range(0,4):
          for w3 in range(0,4):
              wts = [w1/10.,w2/10.,w3/10.]
              wted_preds = np.round(np.tensordot(preds, wts, axes=((0),(0)))).astype('float32')
              df = df.append(pd.DataFrame({'wt1':wts[0],'wt2':wts[1], 'wt3':wts[2], 'IOU': IoU_coef(y_test, wted_preds).numpy() }, index=[0]), ignore_index=True)
  max_iou_row = df.iloc[df['IOU'].idxmax()]
  return [max_iou_row[0], max_iou_row[1], max_iou_row[2]]

def get_ensembled_preds(data, models, weights, augment):
  
  y_pred = []
  
  if augment:
    
    for i in tqdm(range(len(data[0])), desc = "performing augmented ensemble predictions"):
      model_inputs = [augment_image(data[k][i]) for k in range(len(models))]
      aug_preds = []
      for j in range(len(model_inputs[0])):
        individual_model_preds = np.array([models[k].predict(np.expand_dims(model_inputs[k][j], axis=0)) for k in range(len(models))])
        weighted_preds = np.round(np.tensordot(individual_model_preds, weights, axes=((0),(0))))
        aug_preds.append(weighted_preds[0])
      y_pred.append(ensemble_augmented_preds(undo_augmentations(aug_preds)))
  
  else: y_pred = np.round(np.tensordot(data, weights, axes=((0),(0))))

  return np.array(y_pred).astype('float32')  


def get_results(dataset_name, pre_processing, post_processing):

  results = {}
  
  normalized_type = "normalized" if pre_processing else "unnormalized"
  augment_type = "augmented" if post_processing else "unaugmented"
  print("".join(["="]*100))
  print("\nfetching results for {} and {} {}\n".format(normalized_type, augment_type, dataset_name))
  print("".join(["="]*100))
  
  print('loading {} {} dataset...'.format(normalized_type, dataset_name))
  with open(os.path.join(ROOT_DIR,"data", "processed", normalized_type, dataset_name + ".dat"),"rb") as f : dataset = pkl.load(f)
  X_test, y_test = dataset['Testing']['TissueImages'], dataset['Testing']['GroundTruth']
  
  backbones = ["resnet101", "inceptionresnetv2","densenet121"]
  models, X_tests, preds, aug_preds = [], [], [], []
  
  print('loading models...')
  for backbone in backbones:
    preprocess_input =  sm.get_preprocessing(backbone)
    X_tests.append(preprocess_input(X_test))
    weight_file = "-".join(map(str,[normalized_type, dataset_name, "UNet", backbone,"best.h5"]))
    models.append(load_model(os.path.join(ROOT_DIR, "models", "weights", weight_file), custom_objects= custom_objs, compile=False))
  
  print('making individual predictions...')
  for i in range(len(models)):
    
    if post_processing:
      model_preds = []
      for j in range(len(X_tests[i])):
        model_preds.append(ensemble_augmented_preds(undo_augmentations(models[i].predict(augment_image(X_tests[i][j])))))
      aug_preds.append(model_preds)
    
    preds.append(models[i].predict(X_tests[i]))
  
  preds = np.array(preds).astype('float32')
  aug_preds = np.array(aug_preds).astype('float32')

  print('fetching ensemble weights...')
  weights = get_ensemble_weights(y_test, preds)
  print('weights for {} {} dataset are {}'.format(normalized_type, dataset_name, weights))

  print('calculating individual model metrics')
  if post_processing:
    for i in range(len(aug_preds)): results[backbones[i]] = get_metrics(y_test, aug_preds[i])
  else:
    for i in range(len(preds)): results[backbones[i]] = get_metrics(y_test, preds[i])

  print('calculating ensembled model metrics...')
  ensembled_preds = None
  if post_processing: ensembled_preds = get_ensembled_preds(X_tests, models, weights, augment = True)
  else : ensembled_preds = get_ensembled_preds(preds, models, weights, augment = False)

  results["ensemble"] = get_metrics(y_test, ensembled_preds)

  return results