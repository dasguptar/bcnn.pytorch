
# Bilinear ConvNets for Fine-Grained Recognition

This is a [PyTorch](http://pytorch.org/) implementation of Bilinear CNNs as described in the paper [Bilinear CNN Models For Fine-Grained Visual Recognition](http://vis-www.cs.umass.edu/bcnn/) by Tsung-Yu Lin, Aruni Roy Chowdhury, and Subhransu Maji. On the [Caltech-UCSD Birds-200-2011](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) or CUB-200-2011 dataset, for the task of 200 class fine-grained bird species classification, this implementation reaches:

- Accuracy of `84.29%` using the following training regime
  - Train only new bilinear classifier, keeping pre-trained layers frozen
    - Learning rate: 1e0, Weight Decay: 1e-8, Epochs: 55
  - Finetune all pretrained layers as well as bilinear layer jointly
    - Learning rate: 1e-2, Weight Decay: 1e-5, Epochs: 25
  - Common settings for both training runs
    - Optimizer: SGD, Momentum: 0.9, Batch Size: 64, GPUs: 4
- These values are plugged into the config file as defaults
- The original paper reports `84.00%` accuracy on CUB-200-2011 dataset using `VGG-D` pretrained model, which is similar to the `VGG-16` model that this implementation uses.
- Minor differences exist, e.g. no SVM being used, and the L2 normalization is done differently.

## Requirements

- Python (tested on **3.6.9**, should work on **3.5.0** onwards due to typing).
- Other dependencies are in `requirements.txt`
- Currently works with Pytorch 1.1.0, but should work fine with newer versions.

## Usage

The actual model class along with the relevant dataset class and a utility trainer class is packaged into the `bcnn` subfolder, from which the relevant modules can be imported. Dataset downloading and preprocessing is done via a shell script, and a Python driver script is provided to run the actual training/testing loop.

- Use the script `scripts/prepareData.sh` which does the following:
  - **WARNING:** Some of these steps require [GNU Parallel](https://www.gnu.org/software/parallel/), which can be installed [via these methods](https://stackoverflow.com/questions/32093425/installing-gnu-parallel-without-root-permission)
  - Download the [CUB-200-2011 dataset](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) and extract it.
  - Preprocess the dataset, i.e. resizing smaller edge to 512 pixels maintaining aspect ratio.
  - A copy of the dataset is also created where images are cropped to their bounding boxes.
- `main.py` is the actual driver script. It imports relevant modules from the `bcnn` package, and performs the actual pre-training and fine-tuning of the model, and testing it on the test splits. For a list of all command-line arguments, have a look at `config.py`.
  - Model checkpoints are saved to the `ckpt/` directory with the name specified by the command line argument `--savedir`.

If you have a working Python3 environment, simply run the following sequence of steps:

```bash
- bash scripts/prepareData.sh
- pip install -r requirements.txt
- export CUDA_VISIBLE_DEVICES=0,1,2,3
- python main.py --gpus 1 2 3 4 --savedir ./ckpt/exp_test
```

## Notes

- (**Oct 12, 2019**) GPU memory consumption is not very high, which means batch size can be increased. However, that requires changing other hyperparameters such as learning rate.

## Acknowledgements

[Tsung-Yu Lin](https://people.cs.umass.edu/~tsungyulin/) and [Aruni Roy Chowdhury](https://arunirc.github.io/) released the [original implementation](https://bitbucket.org/tsungyu/bcnn/src/master/) which was invaluable in understanding the model architecture.  
[Hao Mood](https://haomood.github.io/homepage/) also released a [PyTorch implementation](https://github.com/HaoMood/bilinear-cnn/) which was critical for finding the right hyperparameters to reach the accuracy reported in the paper.  
As usual, shout-out to the [Pytorch team](https://github.com/pytorch/pytorch#the-team) for the incredible library.

## Contact

[Riddhiman Dasgupta](https://dasguptar.github.io/)  
*Please create an issue or submit a PR if you find any bugs!*

## License

**MIT**
