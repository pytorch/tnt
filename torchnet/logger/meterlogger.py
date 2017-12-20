# Copyright (c) 2017, Kui.
#
# xukui.cs@gmail.com
# Tsinghua Univ.
# Modified at Dec 12 2017
#
import numpy as np
import torch
import torchnet as tnt
from torchnet.logger import VisdomPlotLogger, VisdomLogger

class MeterLogger(object):


    def __init__(self, port=8097, nclass=21, title="DNN"):
	self.nclass = nclass
        self.meter  = {}
        self.meter['time']    = tnt.meter.TimeMeter(None)
        self.meter['loss']    = tnt.meter.AverageValueMeter()
        self.meter['acc']     = tnt.meter.ClassErrorMeter(topk=(1, 5), accuracy=True)
        self.meter['map']     = tnt.meter.mAPMeter()
        self.meter['confmat'] = tnt.meter.ConfusionMeter(nclass, normalized=True)
        self.logger = {'Train':{}, 'Test':{}}
        for mode in ['Train', 'Test']:
            title_pre = title+' '+mode
            self.logger[mode]['loss']    = VisdomPlotLogger('line', port=port, opts={'title': title_pre+' Loss'})
            self.logger[mode]['acc']     = VisdomPlotLogger('line', port=port, opts={'title': title_pre +' Accuracy'})
            self.logger[mode]['map']     = VisdomPlotLogger('line', port=port, opts={'title': title_pre +' mAP'})
            self.logger[mode]['confmat'] = VisdomLogger('heatmap', port=port, opts={'title':  title_pre +' Confusion matrix',
                                                     'columnnames': list(range(nclass)),
                                                     'rownames': list(range(nclass))})
    def printMeter(self, mode,  epoch, i, total):
        print('%s: [%d][%d/%d]\t'
              'Loss %.4f (%.4f)\t'
              'Acc@1 %.3f%% \t'
              'Acc@5 %.3f%% \t'
              'mAP %.4f\t'
              'Time %.2f ' % (mode, epoch, i, total, \
              self.meter['loss'].val, self.meter['loss'].mean, \
              self.meter['acc'].value()[0], \
              self.meter['acc'].value()[1], \
              self.meter['map'].value(), \
              self.meter['time'].value()))
    
    def ver2Tensor(self, target):
        target_mat  = torch.zeros(target.shape[0], self.nclass)
        for i,j in enumerate(target):
            target_mat[i][j]=1
        return target_mat


    def updateMeter(self, output, target, loss=None):
	if loss is not None:
           self.meter['loss'].add(loss.data[0])
        self.meter['acc'].add(output.data,target.cuda(async=True))
        target_th = self.ver2Tensor(target)
        self.meter['map'].add(output.data,target_th.cuda(async=True))
        self.meter['confmat'].add(output.data,target_th.cuda(async=True))
    
    def resetMeter(self, epoch, mode='Train'):
	if ~np.isnan(self.meter['loss'].value()[0]):
           self.logger[mode]['loss'].log(epoch, self.meter['loss'].value()[0])
        self.logger[mode]['acc'].log(epoch, self.meter['acc'].value()[0])
        self.logger[mode]['map'].log(epoch, self.meter['map'].value())
        self.logger[mode]['confmat'].log(self.meter['confmat'].value())
        for met in ['loss', 'acc', 'map', 'confmat']:
            self.meter[met].reset()

