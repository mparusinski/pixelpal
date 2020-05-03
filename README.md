PixelPal
============
[![GitHub stars](https://img.shields.io/github/stars/mparusinski/hdpixels)](https://github.com/mparusinski/hdpixels/stargazers)[![GitHub issues](https://img.shields.io/github/issues/mparusinski/hd-pixels)](https://github.com/mparusinski/hdpixels/issues)[![GitHub forks](https://img.shields.io/github/forks/mparusinski/hd-pixels)](https://github.com/mparusinski/hdpixels/network)[![GitHub license](https://img.shields.io/github/license/mparusinski/hd-pixels)](https://github.com/mparusinski/hdpixels/blob/master/LICENSE.md)

This is a python application powered by Tensorflow and Keras that uses
deep learning to enhance low resolution images like icons.

---

## Setup
Clone this repo to your desktop and setup your virtual environment:

```bash
virtualenv -p /usr/bin/python3 venv
source venv/bin/activation
pip install -r requirements.txt
```

You need to install Tensorflow 2.0 :
* CPU version : `pip install tensorflow`
* GPU (NVIDIA) version : `pip install tensorflow-gpu`
* GPU (AMD) version : `pip install tensorflow-rocm`

> AMD's rocm is still in early development. 
> NVIDIA requires CUDA to be installed.

---

## Usage

You can run `hdpixels` in two ways : either to train a network or to augment images.

### Training

To train a network run
```bash
python hdpixels train 'python.module.path' ./path/to/dataset ./path/to/weights.h5 --validation-dataset ./path/to/validation-dataset --callbacks list_of_python_modules
```

so for instance"
```bash
python hdpixels train 'model.pre_upsampling' data/processed/training/ ./models/preupsampling/v0.h5 --validation-dataset data/processed/validation --callbacks 'callbacks.csv_logger' 'callbacks.early_stopping' 'callbacks.model_checkpoint'
```

### Augmenting

To visualise the run of a network (or not)
```bash
python hdpixels augment /path/to/image.png --module 'python.module.path' --weights ./path/to/weights.h5
```

so for instance:
```bash
python hdpixels augment data/processed/training/32x32/0f31d855-ff12-4a8f-87a1-f06438f85123.png --module 'model.pre_upsampling' ./models/preupsampling/weights.h5
```

## Datasets

Datasets are expected to be in the following folder structure :

```bash
dataset
├── 32x32
│   ├── 0018ad30-f5de-4334-8239-6e7da4cfdf20.png
│   ├── 0042c422-a0cf-451d-bb28-45789f1cd366.png
│   ├── ...
│   └── ffc96005-f114-4810-90cc-6f5bbd85bb9a.png
└── 64x64
    ├── 0018ad30-f5de-4334-8239-6e7da4cfdf20.png
    ├── 0042c422-a0cf-451d-bb28-45789f1cd366.png
    ├── ...
    └── ffc96005-f114-4810-90cc-6f5bbd85bb9a.png

```

To donwload a copy of the default dataset do to the following:

```bash
cd data/raw
./download.sh
cd ../../
python data/processed/prepare.py
```

---

## License
>You can check out the full license [here](https://github.com/mparusinski/hdpixels/blob/master/LICENSE.md)

This project is licensed under the terms of the **2 clauses BSD** license.
