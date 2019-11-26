#coding:utf8
import warnings
import os

class DefaultConfig(object):
    env = 'default' # visdom environment
    model = 'ResNet34' # the model used
    '''
    train_data_root = './data/train/' 
    test_data_root = './data/test1' 
    load_model_path = 'checkpoints/model.pth' # path of trained models
'''
    train_data_root = 'C:\\Users\\邵屹宁\\Desktop\\fake\\'
    test_data_root = './data/test1' 
    load_model_path = 'checkpoints/model.pth' # path of trained models

    batch_size = 32 # batch size
    use_gpu = True # user GPU or not
    num_workers = 4 # how many workers for loading data
    print_freq = 20 # print info every N batch

    debug_file = '/tmp/debug' # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'
      
    max_epoch = 100
    lr = 0.001 # initial learning rate
    lr_decay = 0.95 # when val_loss increase, lr = lr*lr_decay
    weight_decay = 1e-4 # loss function

    os.environ['CUDA_VISIBLE_DEVICES']='0'


def parse(self,kwargs):
        '''
        According to dict kwargs update config parameters
        '''
        for k,v in kwargs.items():
            if not hasattr(self,k):
                warnings.warn("Warning: opt has not attribut %s" %k)
            setattr(self,k,v)

        print('user config:')
        for k,v in self.__class__.__dict__.items():
            if not k.startswith('__'):
                print(k,getattr(self,k))


DefaultConfig.parse = parse
opt =DefaultConfig()
# opt.parse = parse
