# image-compression-with-swin-transformer
Learned image compression with transformers

https://spie.org/Publications/Proceedings/Paper/10.1117/12.2656516?SSO=1
## Citation
```
@inproceedings{shen2023learned,
  title={Learned image compression with transformers},
  author={Shen, Tianma and Liu, Ying},
  booktitle={Big Data V: Learning, Analytics, and Applications},
  volume={12522},
  pages={10--20},
  year={2023},
  organization={SPIE}
}
```

## Installation

This method high depend on [CompressAI](https://github.com/InterDigitalInc/CompressAI), If you meet some problems for install compressai, please check their Doc firstly.
```bash
conda create -n compress python=3.7
conda activate compress
pip install compressai
pip install pybind11
git clone https://github.com/stm233/image-compression-with-swin-transformer image-compression
cd image-compression
pip install -e .
pip install -e '.[dev]'
```

## Usage

### Dataset
| Split | Source | Link |
|-------|--------|------|
| Training set | [Open Images](https://storage.googleapis.com/openimages/web/index.html) training set | [Download](https://storage.googleapis.com/openimages/web/download_v7.html) |
| Test set | [Open Images](https://storage.googleapis.com/openimages/web/index.html) validation set | [Download](https://storage.googleapis.com/openimages/web/download_v7.html) |


#### Data Structure
Please put the training and validation data into the right path, or you need to fix the datasets/utils.py

- rootdir/
    - train/
        - data/   
            - img000.png
            - img001.png
    - test/
        - data/  
            - img000.png
            - img001.png

### Pre-trained Model

[checkpoints](https://drive.google.com/file/d/1tRsx-ek8O2lXlcLdMnQ9q5sD-V_4nuGQ/view?usp=drive_link) 



### Training

Here are some super-parameters you need to set up.

```
-m stf8 # model name, if you have youe own model, you need setup at /z00/__int__.py and /models/ __int__.py
-d /data/Dataset/openimages/ # the path to store your training and validaiton data
--lambda 0.0035 # the number to adjust the bit rate of your model, the lambda is higher, the bit rate is higher
--batch_size 12 # this is depend on your GPU's memory
--patch_szie 256,256 # this is the input image's size, we prefer to enlarge the size when the model convergenced
--save_path ./save/ # the save path for your model
--checkpoint ./save/23.ckpt # if this is empty, the model will be trained from the stratch
```

If you wanna use the default super parameter to train your model, we can miss some items in your command.

Eg.

```
python train.py -m stf8 -d /data/Dataset/openimages/ --lambda 0.025 --batch_size 24
```

### Testing / Evaluation

Use `compressai/utils/eval_model/__main__.py` to evaluate a trained checkpoint on a folder of images. It reports average PSNR, MS-SSIM, and bpp, and can save reconstructed images.

Main arguments:

```
-d /path/to/testset/          # folder of images (.png / .jpg / ...)
-a stf8                       # model architecture
-p ./save/23.ckpt             # checkpoint path
-r ./reconstruction           # where to save reconstructed images
--entropy-estimation          # use entropy estimation (default: True)
--cuda                        # run on GPU if available (default: True)
-v                            # verbose mode
```

Example:

```
python -m compressai.utils.eval_model \
  -d /data/Dataset/openimages/test/data/ \
  -a stf8 \
  -p ./save/23.ckpt \
  -r ./reconstruction \
  -v
```

Or equivalently:

```
python compressai/utils/eval_model/__main__.py \
  -d /data/Dataset/openimages/test/data/ \
  -a stf8 \
  -p ./save/23.ckpt \
  -r ./reconstruction \
  -v
```

