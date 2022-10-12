Project Structure
=======================

The top level modules in TorchTNT are:

1. :mod:`torchtnt.data`: contains data related abstractions and utilities for DataLoader v1. For more expansive support, please see TorchData: https://github.com/pytorch/data
2. :mod:`torchtnt.loggers`: contains wrappers around different publishers to make logging metrics uniform and simple.
3. :mod:`torchtnt.runner`: contains a lightweight training framework to simplify maintaining training, evaluation, and prediction loops.
4. :mod:`torchtnt.utils`: contains a grab-bag of various independent, training-related utilities.
