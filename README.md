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
pip install -e .
pip install -e '.[dev]'
```

## Usage

### Dataset

| Split | Dataset | Download |
|-------|---------|----------|
| Training | [Open Images](https://storage.googleapis.com/openimages/web/index.html) | [Download page](https://storage.googleapis.com/openimages/web/download_v7.html) |
| Test | [Kodak](https://r0k.us/graphics/kodak/) (24 images, `kodim01.png`–`kodim24.png`) | [Download images](https://r0k.us/graphics/kodak/) |
| Test | [Tecnick](https://testimages.org/sampling/) TESTIMAGES SAMPLING (40 images, **1200×1200**, 8-bit RGB, shift **`T01R01`**) | [1200×1200 8BIT RGB pack](https://sourceforge.net/projects/testimages/files/SAMPLING/8BIT/RGB/SAMPLING_8BIT_RGB_1200x1200.tar.bz2/download) |
| Test | [CLIC](https://www.compression.cc/) **2020 professional validation** (41 images, mixed resolution, typically up to 2048 on the long side) | [professional_valid_2020.zip](https://data.vision.ee.ethz.ch/cvl/clic/professional_valid_2020.zip) |

**Tecnick note.** Official reference images are 2400×2400; the pack above is the factor-2 resize to 1200×1200. That archive contains **9 shift folders** (`C00C00`, `T01R01`, `B01C00`, …). We evaluate on the **`T01R01`** subset only (40 PNGs named like `img_1200x1200_3x8bit_T01R01_RGB_*.png`). After download:

```bash
tar -xjf SAMPLING_8BIT_RGB_1200x1200.tar.bz2
# keep only the T01R01 folder (or copy its 40 PNGs into a flat directory for -d)
```

#### Training data structure
Please put the training data into the right path, or you need to fix `datasets/utils.py`.

- rootdir/
    - train/
        - data/   
            - img000.png
            - img001.png

#### Test data
Evaluation (`eval_model`) takes a **flat folder of images** via `-d` (not the train `root/train/data` layout). After downloading, point `-d` to the folder that directly contains the PNG/JPG files, for example:

| Dataset | Example local path after download |
|---------|-----------------------------------|
| Kodak (kodim) | `/data/Dataset/kodim` |
| Tecnick (`T01R01`, 40×1200×1200) | `/data/Dataset/tecnick` |
| CLIC (2020 professional validation, 41 images) | `/data/Dataset/testdata/CLIC/original` |

### Pre-trained Model

[checkpoints](https://huggingface.co/SCU-VIP-Lab/Learned-image-compression-with-transformers/tree/main) 



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

Use `compressai.utils.eval_model` to evaluate a trained checkpoint on a folder of images (Kodak / Tecnick / CLIC). It reports average PSNR, MS-SSIM, and bpp, and can save reconstructed images.

Example (Kodak):

```bash
python -m compressai.utils.eval_model \
  -d /data/Dataset/kodim \
  -a stf8 \
  -p ./checkpoints/save1.ckpt \
  -r ./reconstruction_kodim \
  -v
```

