# PCRLv2 (TPAMI'23)
This repository contains an official implementation of PCRLv2. The accompanying paper "[A Unified Visual Information Preservation Framework for Self-supervised Pre-training in Medical Image Analysis](https://arxiv.org/pdf/2301.00772.pdf)" has been accepted by IEEE TPAMI. 

Note that PCRLv2 is an improved version of PCRLv1 "[Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts](https://arxiv.org/pdf/2109.04379.pdf)".

**If you would like to use our model to do segmentation, please ADD SKIP CONNECTIONS to the UNet architecture! Otherwise, the segmentation performance might be underperformed.**
----
## How to Perform Fine-tuning

----
## on NIH ChestX-ray

### Step 1

Download NIH ChestX-ray from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC) (the same as the first step in the pre-training stage). You also need to download the pre-trained model weights and put them under `pretrained_weight/` as instructed in the [master](https://github.com/RL4M/PCRLv2/tree/main) branch.

### Step 2

```
python main.py --data path_to_chest14 --model pcrlv2 --phase finetune --lr 1e-4 --output ./chest14_finetune_weight --weight ./pretrained_weight/simance_multi_crop_chest_pretask_1.0_240.pt --n chest --d 2 --gpus 0 --ratio 0.8
```

----
## on BraTS'18 (Brain Tumor Segmentation)

### Step 1 

Download the BraTS'18 dataset first. The image folder of BraTS'18 should be organized as follows:

```
BraTS'18/
|--- HGG/
|  	|--- Brats18_TCIA01_460_1/
|	|	|--- Brats18_TCIA01_460_1_t1.nii.gz
|	|	|--- Brats18_TCIA01_460_1_t2.nii.gz
|	|	|--- Brats18_TCIA01_460_1_flair.nii.gz
|	|	|--- Brats18_TCIA01_460_1_t1ce.nii.gz
|  	|--- ...
|--- LGG/
|  	|--- Brats18_TCIA12_466_1/
|	|	|--- Brats18_TCIA12_466_1_t1.nii.gz
|	|	|--- Brats18_TCIA12_466_1_t2.nii.gz
|	|	|--- Brats18_TCIA12_466_1_flair.nii.gz
|	|	|-- -Brats18_TCIA12_466_1_t1ce.nii.gz
|  	|--- ...
```



### Step 2 

```
python main.py --data path_to_brats --model pcrlv2 --phase finetune --lr 1e-4 --output ./brats_finetune_weight --weight ./pretrained_weight/simance_multi_crop_luna_pretask_1.0_240.pt --n brats --d 3 --gpus 0,1,2,3 --b 4 --ratio 1.0
```

Note that Here we segment wt, et, tc simultaneously.

----
## on LiTS'17 (Liver Segmentation)

To achieve better segmentation performance on LiTS dataset, we modify the fine-tuning code provided by [MICCAI-LITS2017](https://github.com/assassint2017/MICCAI-LITS2017).

### Step 1

First download the LiTS17 dataset from [here](https://competitions.codalab.org/competitions/17094). The image folder of LiTS'17 should be organized as follows:

```
LiTS'17/
|--- train/
|	|--- CT/
|	|	|--- volume10.nii.gz
|	|	|--- ...
|	|--- seg/
|	|	|--- segmentation10.nii.gz
|	|	|--- ...
|--- test/
|	|--- CT/
|	|	|--- volume22.nii.gz
|	|	|--- ...
|	|--- seg/
|	|	|--- segmentation22.nii.gz
|	|	|--- ...
```



### Step 2

```
cd MICCAI-LITS2017
```

### Step 3

Change default data paths in `MICCAI-LITS2017/parameter.py` to your own paths where LiTS data is stored. And modity the training_set_path, valid_set_path to the path you want to store your processed LiTS'17 dataset. 

### Step 4

Preprocess the training set.

```
cd data_prepare
python get_training_set.py
```

### Step 5

```
python train_ds.py --weight ../download_weight/simance_multi_crop_luna_pretask_1.0_240.pt --gpus 0,1
```

