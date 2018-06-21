TNT
==========

**TNT** is a library providing powerful dataloading, logging and visualization utlities for Python.
It is closely intergrated with [PyTorch](http://pytorch.org) and is designed to enable rapid iteration with any model 
or training regimen.

![travis](https://travis-ci.org/pytorch/tnt.svg?branch=master) 
[![Documentation Status](https://readthedocs.org/projects/tnt/badge/?version=latest)](http://tnt.readthedocs.io/en/latest/?badge=latest)

- [About](#about)
- [Installation](#installation)
- [Documentation](http://tnt.readthedocs.io)
- [Getting Started](#getting-started)


## Installation

Make sure you have PyTorch installed, then do:

```buildoutcfg
pip install git+https://github.com/pytorch/tnt.git@master
```

To update:

```buildoutcfg
pip install --upgrade git+https://github.com/pytorch/tnt.git@master
```

## About
TNT (imported as _torchnet_) is a framework for PyTorch which provides a set of abstractions for PyTorch 
aiming at encouraging code re-use as well as encouraging modular programming. It provides powerful dataloading, logging,
and visualization utilities. 

The project was inspired by [TorchNet](https://github.com/torchnet/torchnet), and legend says that it stood for “TorchNetTwo”. 
Since the deprecation of TorchNet TNT has developed on its own.

For example, TNT provides simple methods to record model preformance in the `torchnet.meter` module and to log them to Visdom
(or in the future, TensorboardX) with the `torchnet.logging` module.

In the future, TNT will also provide strong support for multi-task learning and transfer learning applications. It 
currently supports joint training data loading through torchnet.utils.MultiTaskDataLoader.

Most of the modules support NumPy arrays as well as PyTorch tensors on input, and so could potentially be used with 
other frameworks.


## Getting Started
See some of the examples in https://github.com/pytorch/examples. We would like to include some walkthroughs in the
[docs](https://tnt.readthedocs.io) (contributions welcome!).


## [LEGACY] Differences with lua version

What's been ported so far:

 * Datasets:
   * BatchDataset
   * ListDataset
   * ResampleDataset
   * ShuffleDataset
   * TensorDataset [new]
   * TransformDataset
 * Meters:
   * APMeter
   * mAPMeter
   * AverageValueMeter
   * AUCMeter
   * ClassErrorMeter
   * ConfusionMeter
   * MovingAverageValueMeter
   * MSEMeter
   * TimeMeter
 * Engines:
   * Engine
 * Logger
   * Logger
   * VisdomLogger
   * MeterLogger [new, easy to plot multi-meter via Visdom]

 Any dataset can now be plugged into `torch.utils.DataLoader`, or called
 `.parallel(num_workers=8)` to utilize multiprocessing.
