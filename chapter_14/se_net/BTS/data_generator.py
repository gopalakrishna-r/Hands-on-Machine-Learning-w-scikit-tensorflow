import itertools
import scipy
import numpy as np
from tensorflow.keras.utils import Sequence, to_categorical
from functional import seq

from fn import _

class CustomDataGenerator(Sequence):
    def __init__(self, hdf5_file, brain_idx, batch_size = 16, view = "axial", mode = 'train', horizontal_flip = False, 
                 vertical_flip = False, rotation_range = 0, zoom_range = 0., shuffle = True):
        self.data_storage = hdf5_file.root.data
        self.truth_storage = hdf5_file.root.truth
        
        total_brains = self.data_storage.shape[0]
        self.brain_idx = self.get_brain_idx(brain_idx, mode, total_brains)
        self.batch_size = batch_size
        
        if view == 'axial':
            self.view_axes = (0, 1, 2, 3)
        elif view == 'sagittal':
            self.view_axes = (2, 1, 0, 3)
        elif view == 'coronal':
            self.view_axes = (1, 2, 0, 3)
        else:
            ValueError(f'unknown input view => {view}')
        
        self.mode            = mode
        self.horizontal_filp = horizontal_flip
        self.veritcal_flip   = vertical_flip
        self.rotation_range  = rotation_range
        self.zoom_range      = zoom_range
        self.shuffle         = shuffle
        self.data_shape      = tuple(np.array(self.data_storage.shape[1:])[self.view_axes])
        
        print(f'Using {len(self.brain_idx)} out of {total_brains} brains')
        print(f'({len(self.brain_idx) * self.data_shape[0]} out of {total_brains * self.data_shape[0]} 2D slices)')
        print(f'the generated data shape in "{view}" view: {str(self.data_shape[1:])}')
        print('-----'*10)

    @staticmethod
    def get_brain_idx(brain_idx, mode, total_brains):
        if mode == 'validation':
            brain_idx = np.array([i for i in range(total_brains) if i not in brain_idx])
        elif mode == 'train':
            brain_idx = brain_idx
        else:
            ValueError(f'unknown mode => {mode}')
        return brain_idx
    
    
    def __len__(self):
        return int(np.floor(len(self.brain_idx) / self.batch_size))
    
    def __getitem__(self, index):
        
        idx = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        
        X_batch , Y_batch = self.data_load_and_preprocess(idx)
        return X_batch, Y_batch
    
    def on_epoch_end(self):
        self.indexes  =  [(i, j) for i in self.brain_idx for j in range(self.data_shape[0])]
        
        if self.mode == 'train' and self.shuffle:
            np.random.shuffle(self.indexes)
    
    def data_load_and_preprocess(self, idx):
        slice_batch = []
        label_batch = []
        
        (seq(idx) 
            .map(lambda index: (index[0], index[1])) 
              .map(self.read_data(_[0], _[1])) 
                .map((self.normalize_modalities(_[0]), _[1])) 
                  .map(np.concatenate((_[0], _[1]), axis = 1)) 
                    .map(self.apply_transform(_, self.get_random_transform())) 
                        .map((_[..., :4], to_categorical(_[..., 4], 4))) 
                            .for_each(lambda slice_and_label: slice_batch.append(slice_and_label[0], label_batch.append(slice_and_label[1]))))
                            
        return np.array(slice_batch), np.array(label_batch)
    
    
    def read_data(self, brain_number, slice_number):
        slice_ = self.data_storage[brain_number].transpose(self.view_axes)[slice_number]
        label_ = self.truth_storage[brain_number].transpose(self.view_axes[:3])[slice_number]     
        label_ = np.expand_dims(label_, axis = -1)   
        
        return slice_, label_
    
    
    def normalize_slice(self, slice):
        b = np.percentile(slice, 99)
        t = np.percentile(slice, 1)
        slice = np.clip(slice, t, b)

        if np.std(slice) != 0:
            slice = (slice - np.mean(slice)) / np.std(slice)
        return slice
    
    def normalize_modalities(self, Slice):
        
        normalized_slices = np.zeros_like(Slice).astype(np.float32)
        for slice_ix in range(4):
            normalized_slices[..., slice_ix] = self.normalize_slice(Slice[..., slice_ix])

        return normalized_slices
    
    def flip_axis(self, x, axis):
        x = np.asarray(x).swapaxes(axis, 0)
        x = x[::-1, ...]
        x = x.swapaxes(0, axis)
        return x
