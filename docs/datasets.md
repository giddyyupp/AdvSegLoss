## Datasets
We used below four datasets in our experiments.

### Bedroom Dataset
1355 train and 135 test bedroom images from [Ade20k Dataset](http://sceneparsing.csail.mit.edu/). [[Citation](../datasets/bibtex/ade20k.tex)]

### Illustraion Dataset
To download illustration images, please refer to [Ganilla](https://github.com/giddyyupp/ganilla). [[Citation](../datasets/bibtex/ganilla.tex)]

### Cityscapes Dataset
2975 train and 500 test images from the [Cityscapes training set](https://www.cityscapes-dataset.com). [[Citation](../datasets/bibtex/cityscapes.tex)]

### Coco Datasets

We shared coco elephant and sheep datasets in this [GDrive folder](https://drive.google.com/drive/folders/15osbtUQxLyG_EnO7HHq4rBaoLU0BkMpP?usp=sharing)

### Your Own Datasets
To train a model on your own datasets, you need to create a data folder with two subdirectories `trainA` and `trainB` that contain images from domain A and B. 
You can test your model on your training set by setting `--phase train` in `test.py`. You can also create subdirectories `testA` and `testB` if you have test data.

### Paired Data Preparation
We provide a python script to generate pix2pix training data in the form of pairs of images {A,B}, where A and B are two different depictions of the same underlying scene. For example, these might be pairs {label map, photo} or {bw image, color image}. Then we can learn to translate A to B or B to A:

Create folder `/path/to/data` with subfolders `A` and `B`. `A` and `B` should each have their own subfolders `train`, `test`, etc. In `/path/to/data/A/train`, put training images in style A. In `/path/to/data/B/train`, put the corresponding images in style B. Repeat same for other data splits (`test`, etc).

Corresponding images in a pair {A,B} must be the same size and have the same filename, e.g., `/path/to/data/A/train/1.jpg` is considered to correspond to `/path/to/data/B/train/1.jpg`.

Once the data is formatted this way, call:
```bash
python datasets/combine_A_and_B.py --fold_A /path/to/data/A --fold_B /path/to/data/B --fold_AB /path/to/data
```

This will combine each pair of images (A,B) into a single image file, ready for training.


### Edge Map Preparation 

In the `scripts/hed/edges` folder, we provide edge map extraction scripts.

- First run `batch_hed.py`. Required steps and explanations are given in the top of that script.
- Then, run `postprocess_main.m`. Again explanations are given in the top of that script.

Repeat that procedure for trainA and testA folders.
