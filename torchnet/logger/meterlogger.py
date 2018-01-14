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


    def __init__(self, server="http://localhost", port=8097, nclass=21, title="DNN"):
	self.nclass = nclass
        self.meter  = {}
        self.timer    = tnt.meter.TimeMeter(None)
	self.server = server
	self.port = port
	self.nclass = nclass
	self.topk = 5 if nclass > 5 else nclass
	self.title = title
        self.logger = {'Train':{}, 'Test':{}}
    
    def __ver2Tensor(self, target):
        target_mat  = torch.zeros(target.shape[0], self.nclass)
        for i,j in enumerate(target):
            target_mat[i][j]=1
        return target_mat

    def __toTensor(self, var):
	if isinstance(var, torch.autograd.Variable):
		var = var.data
	if not torch.is_tensor(var):
		var = torch.from_numpy(var)
	return var
	
    
    def __addlogger(self, meter, ptype):
	if ptype == 'line':
	   opts={'title': self.title+' Train '+ meter}
	   self.logger['Train'][meter]   = VisdomPlotLogger(ptype,server=self.server, port=self.port, opts=opts)
	   opts={'title': self.title+' Test '+ meter}
	   self.logger['Test'][meter]    = VisdomPlotLogger(ptype,server=self.server, port=self.port, opts=opts)
	elif ptype == 'heatmap':
	   names = list(range(self.nclass))
	   opts={'title': self.title+' Train '+ meter, 'columnnames':names, 'rownames': names }
	   self.logger['Train'][meter]   = VisdomLogger('heatmap', server=self.server, port=self.port, opts=opts)
	   opts={'title': self.title+' Test '+ meter, 'columnnames':names, 'rownames': names }
	   self.logger['Test'][meter]    = VisdomLogger('heatmap', server=self.server, port=self.port, opts=opts)
	
    def __addloss(self, meter):
	self.meter[meter]    = tnt.meter.AverageValueMeter()
	self.__addlogger(meter, 'line')

    def __addmeter(self, meter):
	if meter == 'accuracy':
	    self.meter[meter]    = tnt.meter.ClassErrorMeter(topk=(1, self.topk), accuracy=True)
	    self.__addlogger(meter, 'line')
	elif meter == 'map':
	    self.meter[meter]    = tnt.meter.mAPMeter()
	    self.__addlogger(meter, 'line')
	elif meter == 'confusion':
	    self.meter[meter]    = tnt.meter.ConfusionMeter(nclass, normalized=True)
	    self.__addlogger(meter, 'heatmap')

    def updateMeterList(self, output, target, meters={'accuracy'}):
	output = self.__toTensor(output)
	target = self.__toTensor(target)
	for meter in meters:
	    if not self.meter.has_key(meter):
	        self.__addmeter(meter)
            if meter in ['ap', 'map', 'confusion']:
               target_th = self.__ver2Tensor(target)
               self.meter[meter].add(output, target_th)
	    else:
               self.meter[meter].add(output, target)


    def updateLoss(self, loss, meter='loss'):
	loss = self.__toTensor(loss)
	if not self.meter.has_key(meter):
            self.__addloss(meter)
        self.meter[meter].add(loss[0])

    def resetMeter(self, epoch, mode='Train'):
	self.timer.reset()
	for key in self.meter.keys():
	    val = self.meter[key].value() 
	    val = val[0] if isinstance(val, (list, tuple)) else val
	    self.logger[mode][key].log(epoch, val)
	    self.meter[key].reset()
	
    def printMeter(self, mode,  epoch, i, total, meterlist=None):
	pstr = "%s: [%d][%d/%d] \t"
	tval = []
	tval.extend([mode, epoch, i, total])
	if meterlist==None:
	    meterlist = self.meter.keys()
	for meter in meterlist:
	    if meter == 'confusion':
		continue
	    if meter == 'accuracy':
		pstr += "Acc@1 %.2f%% \t Acc@"+str(self.topk)+" %.2f%% \t"
		tval.extend([self.meter[meter].value()[0], self.meter[meter].value()[1]])
	    elif meter == 'map':
		pstr += "mAP %.3f \t"
		tval.extend([self.meter[meter].value()])
	    else :
		pstr += meter+" %.3f (%.3f)\t"
		tval.extend([self.meter[meter].val, self.meter[meter].mean])
	pstr += " %.2fs/its\t"
	tval.extend([self.timer.value()])
	print(pstr % tuple(tval))
