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

| Split | Dataset | Notes |
|-------|---------|-------|
| Training | [Open Images](https://storage.googleapis.com/openimages/web/index.html) | [Download page](https://storage.googleapis.com/openimages/web/download_v7.html) |
| Test | [Kodak](https://r0k.us/graphics/kodak/) | 24 images (`kodim01.png`–`kodim24.png`) |
| Test | [Tecnick](https://testimages.org/sampling/) TESTIMAGES SAMPLING | 40 images, **1200×1200**, 8-bit RGB, shift **`T01R01`** |
| Test | [CLIC](https://www.compression.cc/) professional validation | 41 images, mixed resolution (folder name on HF: `clic2021_valid`) |

We host the three test sets on Hugging Face: [SCU-VIP-Lab/compression-eval-datasets](https://huggingface.co/datasets/SCU-VIP-Lab/compression-eval-datasets).

```bash
hf download SCU-VIP-Lab/compression-eval-datasets --repo-type dataset --local-dir ./compression-eval-datasets
```

Layout after download:

```
compression-eval-datasets/
  kodak/           # 24 images
  tecnick/         # 40 images (T01R01, 1200×1200)
  clic2021_valid/  # 41 images (CLIC 2020 professional validation)
```

#### Training data structure
Please put the training data into the right path, or you need to fix `datasets/utils.py`.

- rootdir/
    - train/
        - data/   
            - img000.png
            - img001.png

#### Test data
Evaluation (`eval_model`) takes a **flat folder of images** via `-d` (not the train `root/train/data` layout). After downloading from Hugging Face, point `-d` to the corresponding folder:

| Dataset | Path after `hf download` |
|---------|--------------------------|
| Kodak | `./compression-eval-datasets/kodak` |
| Tecnick (`T01R01`) | `./compression-eval-datasets/tecnick` |
| CLIC (professional validation) | `./compression-eval-datasets/clic2021_valid` |

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
  -d ./compression-eval-datasets/kodak \
  -a stf8 \
  -p ./checkpoints/save1.ckpt \
  -r ./reconstruction_kodim \
  -v
```

Tecnick / CLIC: change `-d` to `./compression-eval-datasets/tecnick` or `./compression-eval-datasets/clic2021_valid`.
