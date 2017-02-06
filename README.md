PyTorchNet
==========

PyTorch version of https://github.com/torchnet/torchnet

![travis](https://travis-ci.org/pytorch/tnt.svg?branch=master)

_torchnet_ is a framework for torch which provides a set of abstractions aiming
at encouraging code re-use as well as encouraging modular programming.

Most of the modules support NumPy arrays as well as PyTorch tensors on input,
so could potentially be used with other frameworks.

## Installation

Make sure you have PyTorch installed, then do:

```buildoutcfg
pip install git+https://github.com/pytorch/tnt.git@master
```

## Differences with lua version

What's been ported so far:

 * Datasets:
   * BatchDataset
   * ListDataset
   * ResampleDataset
   * ShuffleDataset
   * TensorDataset [new]
   * TransformDataset
 * Meters:
   * AverageValueMeter
   * ClassErrorMeter
   * ConfusionMeter
   * TimeMeter
 * Engines:
   * Engine
 * Transforms
 
 Any dataset can now be plugged into `torch.utils.DataLoader`, or called
 `.parallel(num_workers=8)` to utilize multiprocessing.
