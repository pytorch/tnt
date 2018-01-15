# mnist_with_meterlogger



## Start Visdom on a server

```bash
python -m visdom.server
# python -m visdom.server -port 9999 # to specify port to ex, 9999
```


## Run Example

```bash
python mnist_with_meterlogger.py
# CUDA_VISIBLE_DEVICES=1 python mnist_with_meterlogger.py # to specify GPU id to ex. 1
```

## Multi-meter

Easy to plot multi-meter with just one-line code:

### Plotting Accuracy, mAP

```python
mlog.updateMeter(output, target, meters={'accuracy', 'map'})
```

### Plotting Loss Curve

```python
# NLL Loss
nll_loss = F.nll_loss(output, target)
mlog.updateLoss(nll_loss, meter='nll_loss')

# Cross Entropy Loss
ce_loss = F.cross_entropy(output, target)
mlog.updateLoss(ce_loss, meter='ce_loss')
```

## Remote Plotting

```python
mlog = MeterLogger(server="Server's IP", nclass=10, title="mnist")
```


## Figure 

![visdom.png](meterlogger.png)
