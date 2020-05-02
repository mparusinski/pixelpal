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

---

## License
>You can check out the full license [here](https://github.com/mparusinski/hdpixels/blob/master/LICENSE.md)

This project is licensed under the terms of the **2 clauses BSD** license.
