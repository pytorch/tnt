.. role:: hidden
    :class: hidden-section

torchnet.engine
======================

.. automodule:: torchnet.engine
.. currentmodule:: torchnet.engine

Engines are a utility to wrap a training loop. They provide several hooks which 
allow users to define their own fucntions to run at specified points during the 
train/val loop.

Some people like engines, others do not. TNT is build modularly, so you can use
the other modules with/without using an engine. 

torchnet.engine.Engine
-----------------------------

.. autoclass:: Engine
    :members:
    :undoc-members:
    :show-inheritance:

