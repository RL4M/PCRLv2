# PCRLv2 (TPAMI'23)
This repository contains an official implementation of PCRLv2. The accompanying paper "[A Unified Visual Information Preservation Framework for Self-supervised Pre-training in Medical Image Analysis](https://arxiv.org/pdf/2301.00772.pdf)" has been accepted by IEEE TPAMI. 

Note that PCRLv2 is an improved version of PCRLv1 "[Preservational Learning Improves Self-supervised Medical Image Models by Reconstructing Diverse Contexts](https://arxiv.org/pdf/2109.04379.pdf)".

## Dependencies
Please make sure your PyTorch version >=1.1 before you run the code. We strongly recommend you to install Anaconda3 where we use Python 3.6. In addition, we use [apex](https://github.com/NVIDIA/apex) for acceleration. We also use [pretrained-models.pytorch](https://github.com/Cadene/pretrained-models.pytorch) and [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) to speed up the implementation. 


## How to Load Pretrained Models (2D & 3D)
First, git clone our repository:
``` python
git clone https://github.com/RL4M/PCRLv2.git
cd PCRLv2
mkdir pretrained_weight
```

Download the pretrained model weights from https://drive.google.com/drive/folders/1Rp7CJblA5HzX5xjc_hUEcD2kWFEiTboF?usp=sharing. Put them under the ``pretrained_weight/`` folder:

```
pretrained_weight/
	simance_multi_crop_luna_pretask_1.0_240.pt
	simance_multi_crop_chest_pretask_1.0_240.pt
	simance_multi_crop_chexpter_pretask_1.0_240.pt
```
Here, ``simance_multi_crop_luna_pretask_1.0_240.pt``, ``simance_multi_crop_chest_pretask_1.0_240.pt``, and ``simance_multi_crop_chexpter_pretask_1.0_240.pt`` correspond to pre-trained models on LUNA, NIH ChestX-ray, and CheXpert, respectively.


### Load the Encoder Part of a 2D Model

```python
import segmentation_models_pytorch as smp
aux_params = dict(
            pooling='avg',  # one of 'avg', 'max'
            dropout=0.2,  # dropout ratio, default is None
            activation='sigmoid',  # activation function, default is None
            classes=n_class,  # define number of output labels
        )
weight_path = './pretrained_weight/simance_multi_crop_chexpter_pretask_1.0_240.pt'
model = smp.Unet('resnet18', in_channels=3, aux_params=aux_params, classes=1, encoder_weights=None)
encoder_dict = torch.load(weight_path)['state_dict']
encoder_dict['fc.bias'] = 0
encoder_dict['fc.weight'] = 0
model.encoder.load_state_dict(encoder_dict)
```

### Load the Encoder Part of a 3D Model

```python
from models.pcrlv2_model_3d import PCRLv23d
model = PCRLv23d()
weight_path = './pretrained_weight/simance_multi_crop_luna_pretask_1.0_240.pt'
model_dict = torch.load(weight_path)['state_dict']
model.load_state_dict(model_dict)
```

## How to Perform Pre-training

### on NIH ChestX-ray

#### **Step 1**

Download NIH ChestX-ray from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC). The image folder of NIH ChestX-ray should be organized as follows:
``` python
Chest14/
	images/
		00002639_006.png
		00010571_003.png
		...
```

Besides, we also provide the list of images used for pre-training: ``train_val_txt/chest_train.txt``. Specifically, for semi-supervised experiments, we use top K% images for pre-training and last (100-K)% for fine-tuning. For instance, given a ratio of 9.5:0.5, we use the first 95% images in ``chest_train.txt`` for pre-training, while the rest 5% images are used for fine-tuning.


#### **Step 2**
For pre-training on NIH ChestX-ray, you can run the following script:
``` python
python main.py --data path_to_chest14 --model pcrlv2 --b 64 --epochs 240 --lr 1e-2 --output  saved_dir --n chest --d 2 --gpus 0,1,2,3 --ratio 1.0 --amp
```
or

```
bash run2d.sh
```

``--data`` defines the path where you store NIH ChestX-ray.

``--d`` defines the type of dataset, ``2`` stands for 2D while ``3`` denotes 3D.

``--n`` gives the name of dataset.

``--ratio`` determines the percentage of images in the training set used for pre-training. Specifically, ``1`` means using all training images in the training set to for pre-training.

`--amp` denotes whether you want to use ``apex`` to accelerate the training process.

*Following a similar protocol, you can also perform pre-training on CheXpert.*

### on LUNA

#### **Step 1**

Download LUNA16 from [here](https://luna16.grand-challenge.org/Download/). The image folder of LUNA16 should be organized as follows:
```python
LUNA16/
	subset0/   		   	
		1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.raw
		1.3.6.1.4.1.14519.5.2.1.6279.6001.979083010707182900091062408058.mhd
  	...
	subset1/
	subset2/
	...
	subset9/
```

We provide the list of training images in ``train_val_txt/luna_train.txt``. Similar to NIH ChestX-ray, we used ``luna_train.txt`` for semi-supervised experiments, where top K% images in ``luna_train.txt`` are used for pre-training, and the rest are for fine-tuning.

#### **Step 2**
First, it is suggested to pre-process the LUNA dataset to get cropped pairs from 3D images.

``` python
python luna_preprocess.py --input_rows 64 --input_cols 64 --input_deps 32 --data path_to_LUNA --save path_to_processedLUNA
```

Then, we can start pre-training.
#### **Step 3**
``` python
python main.py --data path_to_processedLUNA  --model pcrlv2 --b 32 --epochs 240 --lr 1e-3 --output saved_dir  --n luna --d 3 --gpus 0,1,2,3 --ratio 1.0 --amp
```
or

```
bash run3d.sh
```



## How to Perform Finetuning

### on NIH ChestX-ray

#### Step 1

Download NIH ChestX-ray from [here](https://nihcc.app.box.com/v/ChestXray-NIHCC) (the same as the first step in chest pretraining).


#### Step 2

```
python main.py --data path_to_chest14 --model pcrlv2 --phase finetune --lr 1e-4 --output ./chest14_finetune_weight --weight ./pretrained_weight/simance_multi_crop_chest_pretask_1.0_240.pt --n chest --d 2 --gpus 0 --ratio 0.8
```



### on Brats18 Brain Tumor Segmentation Challenge Dataset

#### Step1 

Download the brats18 dataset.

#### Step 2 

```
python main.py --data path_to_brats --model pcrlv2 --phase finetune --lr 1e-4 --output ./brats_finetune_weight --weight ./pretrained_weight/simance_multi_crop_luna_pretask_1.0_240.pt --n brats --d 3 --gpus 0,1,2,3 --b 4 --ratio 1.0
```

 Here we segment wt, et, tc simultaneously.



### on LiTS Tumor Segmentation Challenge Dataset

To achieve state-of-art performance on Lits dataset, we modify the finetune code based on [this](https://github.com/assassint2017/MICCAI-LITS2017).

#### Step 1

Download the LiTS17 dataset from [here](https://competitions.codalab.org/competitions/17094).

#### Step 2

```
cd MICCAI-LITS2017
```

#### Step 3

Modify the config file in parameter.py file. Change the data path to your own Lits data path.

#### Step 4

```
python train_ds.py --weight ../download_weight/simance_multi_crop_luna_pretask_1.0_240.pt --gpus 0,1
```

