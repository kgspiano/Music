import os
import pandas as pd
import numpy as np
import gzip

from torchvision.datasets import VisionDataset
from scipy.io import wavfile

class AudioSet(VisionDataset):
    def __init__(self, root, transform = None, target_transform = None, train=True):
        super(AudioSet, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.train = train
        if self.train:
            df = pd.read_csv(os.path.join(root, 'balanced_train_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'train')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
        else:
            df = pd.read_csv(os.path.join(root, 'eval_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'valid')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
            
        _class_labels_indices = pd.read_csv(os.path.join(root, 'class_labels_indices.csv'))
        _label2index = {}
        for row in _class_labels_indices[['mid', 'index']].iterrows():
            _label2index[row[1]['mid']] = row[1]['index']
            
        def convert(x):
            s = x[1:-1]
            labels = s.split(',')
            return np.array([_label2index[l] for l in labels])
        
        df[3] = df[3].map(convert)
        
        self.sample_rate = 22050
        self.paths = []
        for x, y in zip(df[0].to_list(), df[1].to_list()):
            self.paths.append('{yid}_{start:.3f}.wav.gz'.format(yid=x, start=y))

        self.targets = df[3].to_list()
        
        _idx2class = {}
        for row in _class_labels_indices[['index', 'display_name']].iterrows():
            _idx2class[row[1]['index']] = row[1]['display_name']
        self.idx2class = _idx2class
        self.n_class = 527
        
    def loader(self, path):
        with gzip.open(os.path.join(self._dir, path), "rb") as f:
            _, x = wavfile.read(f)
        if len(x.shape) > 1:
            return np.copy(x[:, 0])
        return np.copy(x)
    
    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]
        sample = self.loader(path)
        return sample, target
    
    def __len__(self):
        return len(self.paths)
    
    
# AudioSet2 : option - delete data with some label
class AudioSet2(VisionDataset):
    def __init__(self, root, transform = None, target_transform = None, train=True, deleteidx = None, deleteid = None):
        super(AudioSet2, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.train = train
        if self.train:
            df = pd.read_csv(os.path.join(root, 'balanced_train_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'train')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
        else:
            df = pd.read_csv(os.path.join(root, 'eval_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'valid')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
            
        _class_labels_indices = pd.read_csv(os.path.join(root, 'class_labels_indices.csv'))
        _label2index = {}
        for row in _class_labels_indices[['mid', 'index']].iterrows():
            _label2index[row[1]['mid']] = row[1]['index']
            
        def convert(x):
            s = x[1:-1]
            labels = s.split(',')
            return np.array([_label2index[l] for l in labels])
        
        df[3] = df[3].map(convert)
        
        self.sample_rate = 22050
        self.paths = []
        for x, y in zip(df[0].to_list(), df[1].to_list()):
            self.paths.append('{yid}_{start:.3f}.wav.gz'.format(yid=x, start=y))

        self.targets = df[3].to_list()
        
#         self.paths = [self.paths[idx] for idx in range(len(self.paths)) if 0 in self.targets[idx]]

        if deleteidx is not None:
            tmp = []
            tmp2 = []
            for idx in range(len(self.paths)):

                if not any([x in self.targets[idx] for x in deleteidx]): #(deleteidx not in self.targets[idx]):
                    tmp.append(self.paths[idx])
                    tmp2.append(self.targets[idx])

            self.paths = tmp
            self.targets = tmp2
        
        if deleteid is not None:
            tmp = []
            tmp2 = []
            for idx, idtmp in enumerate(self.paths):
                if not (idtmp in deleteid):
                    tmp.append(self.paths[idx])
                    tmp2.append(self.targets[idx])
            self.paths = tmp
            self.targets = tmp2  
                
                
                
        _idx2class = {}
        for row in _class_labels_indices[['index', 'display_name']].iterrows():
            _idx2class[row[1]['index']] = row[1]['display_name']
        self.idx2class = _idx2class
        self.n_class = 527
        
    def loader(self, path):
        with gzip.open(os.path.join(self._dir, path), "rb") as f:
            _, x = wavfile.read(f)
        if len(x.shape) > 1:
            return np.copy(x[:, 0])
        return np.copy(x)
    
    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]
        sample = self.loader(path)
        return sample, target
    
    def __len__(self):
        return len(self.paths)

    
# AudioSet_shuffle : randomly shuffle label
class AudioSet_shuffle(VisionDataset):
    def __init__(self, root, transform = None, target_transform = None, train=True, deleteidx = None, deleteid = None):
        super(AudioSet_shuffle, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.train = train
        if self.train:
            df = pd.read_csv(os.path.join(root, 'balanced_train_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'train')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
        else:
            df = pd.read_csv(os.path.join(root, 'eval_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'valid')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
            
        _class_labels_indices = pd.read_csv(os.path.join(root, 'class_labels_indices.csv'))
        _label2index = {}
        for row in _class_labels_indices[['mid', 'index']].iterrows():
            _label2index[row[1]['mid']] = row[1]['index']
            
        def convert(x):
            s = x[1:-1]
            labels = s.split(',')
            return np.array([_label2index[l] for l in labels])
        
        df[3] = df[3].map(convert)
        
        self.sample_rate = 22050
        self.paths = []
        for x, y in zip(df[0].to_list(), df[1].to_list()):
            self.paths.append('{yid}_{start:.3f}.wav.gz'.format(yid=x, start=y))

        self.targets = df[3].to_list()
        

#         self.paths = [self.paths[idx] for idx in range(len(self.paths)) if 0 in self.targets[idx]]
        ## remove deleteidx
        if deleteidx is not None:
            tmp = []
            tmp2 = []
            for idx in range(len(self.paths)):

                if not any([x in self.targets[idx] for x in deleteidx]): #(deleteidx not in self.targets[idx]):
                    tmp.append(self.paths[idx])
                    tmp2.append(self.targets[idx])

            self.paths = tmp
            self.targets = tmp2
        
        if deleteid is not None:
            tmp = []
            tmp2 = []
            for idx, idtmp in enumerate(self.paths):
                if not (idtmp in deleteid):
                    tmp.append(self.paths[idx])
                    tmp2.append(self.targets[idx])
            self.paths = tmp
            self.targets = tmp2  
            
        for i, t in enumerate(self.targets):
            t[:] = np.random.randint(527, size = len(t))

        
        _idx2class = {}
        for row in _class_labels_indices[['index', 'display_name']].iterrows():
            _idx2class[row[1]['index']] = row[1]['display_name']
        self.idx2class = _idx2class
        self.n_class = 527
        
    def loader(self, path):
        with gzip.open(os.path.join(self._dir, path), "rb") as f:
            _, x = wavfile.read(f)
        if len(x.shape) > 1:
            return np.copy(x[:, 0])
        return np.copy(x)
    
    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]
        sample = self.loader(path)
        return sample, target
    
    def __len__(self):
        return len(self.paths)


# AudioSet3 : option - delete data with some label & target is binary (e.g. whether input is speech or not speech)
class AudioSet3(VisionDataset):
    def __init__(self, root, transform = None, target_transform = None, train=True, deleteidx = None, useidx = None):
        super(AudioSet3, self).__init__(root, transform=transform,
                                            target_transform=target_transform)
        self.train = train
        if self.train:
            df = pd.read_csv(os.path.join(root, 'balanced_train_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'train')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
        else:
            df = pd.read_csv(os.path.join(root, 'eval_segments.csv'), 
                             skiprows=3, sep=', ', usecols=[0, 1, 3], header=None, quotechar='"', engine='python')
            self._dir = os.path.join(root, 'valid')
            _file_names = [x[:11] for x in os.listdir(self._dir)]
            df = df[df[0].isin(_file_names)]
            
        _class_labels_indices = pd.read_csv(os.path.join(root, 'class_labels_indices.csv'))
        _label2index = {}
        for row in _class_labels_indices[['mid', 'index']].iterrows():
            _label2index[row[1]['mid']] = row[1]['index']
            
        def convert(x):
            s = x[1:-1]
            labels = s.split(',')
            return np.array([_label2index[l] for l in labels])
        
        df[3] = df[3].map(convert)
        
        self.sample_rate = 22050
        self.paths = []
        for x, y in zip(df[0].to_list(), df[1].to_list()):
            self.paths.append('{yid}_{start:.3f}.wav.gz'.format(yid=x, start=y))

        self.targets = df[3].to_list()
#         self.paths = [self.paths[idx] for idx in range(len(self.paths)) if 0 in self.targets[idx]]

        tmp = []
        tmp2 = []
        for idx in range(len(self.paths)):
            if any([x not in self.targets[idx] for x in deleteidx]): #(deleteidx not in self.targets[idx]):
                tmp.append(self.paths[idx])
                tmp2.append(self.targets[idx])
        
        self.paths = tmp
        self.targets = tmp2
        
        # binary classification
        tmp2 = []
        for idx in range(len(self.paths)):
            if any([x in self.targets[idx] for x in useidx]): #(deleteidx not in self.targets[idx]):
                tmp2.append([1])
            else:
                tmp2.append([0])
        self.targets = tmp2
        
        _idx2class = {}
        for row in _class_labels_indices[['index', 'display_name']].iterrows():
            _idx2class[row[1]['index']] = row[1]['display_name']
        self.idx2class = _idx2class
        self.n_class = 2
        
    def loader(self, path):
        with gzip.open(os.path.join(self._dir, path), "rb") as f:
            _, x = wavfile.read(f)
        if len(x.shape) > 1:
            return np.copy(x[:, 0])
        return np.copy(x)
    
    def __getitem__(self, index):
        path = self.paths[index]
        target = self.targets[index]
        sample = self.loader(path)
        return sample, target
    
    def __len__(self):
        return len(self.paths)
    
    
    
    
    
