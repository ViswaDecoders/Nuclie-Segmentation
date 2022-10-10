# nuclei-segmentation

## Aim
The approach proposed in this project uses an ensemble of U-Nets with various CNN architectures as encoder backbones together with stain normalization and test time augmentation to segment nuclei from histopathology image data. 
The proposed model was trained and tested on both single-organ and multi-organ dataset, yielding an F1 score of 84.11%, mean IoU of 81.67%, dice score of 84.11%, accuracy of 92.58% and precision of 83.78% on the multi-organ dataset, and an F1 score of 87.04%, mean IoU of 86.66%, dice score of 87.04%, accuracy of 96.69% and precision of 87.57% on the single-organ dataset.

## Datasets used
  
The primary dataset used is a multi-organ dataset called the MOSID dataset which contains a diverse set of 30 images of H&E-stained images of different organs such as breast, liver, kidney, prostate, bladder, colon and stomach.

![image](https://user-images.githubusercontent.com/63601038/179460823-61ef7ce1-4ac0-4c61-98a9-f1c0fcdfae77.png)

In addition to this, the model was also trained and tested on a secondary single-organ dataset called the TNBC dataset, which contains a number of annotated breast tissue images.

![image](https://user-images.githubusercontent.com/63601038/179460851-2480ca16-fa46-4bf2-aba1-7a1d463c7f02.png)

## Proposed Method

the given histopathology image is first stain normalized using the stain normalizer proposed by (Vahadane et al., 2016). Patches of size 256*256 are then extracted from this stain normalized image and fed into a U-Net ensemble model. This ensemble is a combination of three U-Nets built using different encoder backbones namely, ResNet101, InceptionResNetV2 and Densenet121. Prior to segmentation (model prediction), each of the patches undergo a series of augmentations (as part of the test-time augmentation). Each of these augmented patches is then fed into the ensemble model which returns a prediction mask. The masks of the augmented patches are then merged to obtain the final patch prediction. Once the model has predicted the masks for all patches in a similar manner, the patch masks are merged to reconstruct the final nuclei segmented mask of the given histopathology image.

![image](https://user-images.githubusercontent.com/63601038/179461133-2087eaef-79d0-4c43-9ed5-adec9080e052.png)

