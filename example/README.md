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

## Remote Plotting

```python
mlog = MeterLogger(server="Server's IP", nclass=10, title="mnist")
```


## Figure 

![visdom.png](meterlogger.png)
