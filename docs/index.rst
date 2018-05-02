.. TNT documentation master file, created by
   sphinx-quickstart on Tue May  1 11:04:29 2018.


TNT Documentation
=================================

TNT is a library providing powerful dataloading, logging and visualization utlities for Python.
It is closely intergrated with `PyTorch <http://pytorch.org>`_ and is designed to enable rapid iteration with any model or training regimen.



.. toctree::
   :maxdepth: 1
   :caption: Notes

    Examples <https://github.com/pytorch/tnt/tree/master/example>


.. toctree::
   :maxdepth: 1
   :caption: Package Reference

    torchnet.dataset <source/torchnet.dataset.rst>
    torchnet.engine <source/torchnet.engine.rst>
    torchnet.logger <source/torchnet.logger.rst>
    torchnet.meter <source/torchnet.meter>
    torchnet.utils <source/torchnet.utils>


TNT was inspired by TorchNet, and legend says that it stood for "TorchNetTwo". Since then, TNT has developed
on its own.

TNT provides simple methods to record model preformance in the `torchnet.meter <source/torchnet.meter>`_ module
and to log them to Visdom (or in the future, TensorboardX) with the `torchnet.logging <source/torchnet.logging>`_ 
module.

In the future, TNT will also provide strong support for multi-task learning and transfer learning applications. It
currently supports joint training data loading through 
`torchnet.utils.MultiTaskDataLoader <source/torchnet.utils.html#multitaskdataloader>`_.

