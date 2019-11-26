#coding:utf8
import torch as t
import time


class BasicModule(t.nn.Module):
    '''
    envelopped nn.Module,mainly provides two methods: save and load
    '''

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# default name

    def load(self, path):
        '''
        capable to load model with specified path
        '''
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        '''
        save module, defaultly use module-name and time as filename
        '''
        if name is None:
            #prefix = 'checkpoints/' + self.model_name + '_'
            prefix = 'J:\\checkpoints' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H_%M_%S.pth')
        t.save(self.state_dict(), name)
        return name


class Flat(t.nn.Module):
    '''
    reshape the input to （batch_size,dim_length）
    '''

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)
