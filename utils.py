import numpy as np
import torch
from torch.utils.data import Dataset
import pywt
import pandas as pd
import tqdm
from typing import List


class Decomposer():
    '''
    Base Decoder
    Provides wavedec for a given signal, concatenates dec. levels and flattens them 
    into a single vector dropping NAs. 
    '''
    def __init__(self, maxcurrent = 500):
        self.max_current = maxcurrent
        pass
    
    @staticmethod
    def _toAmperes(data):
        return data/1e3
    
    def _normalize_by_current(self, data):
        return data / (self.max_current * np.sqrt(2))
    
    def normalize(self, batch: list):
        '''
        Takes raw batch and returns it normalized in amperes
        '''
        result = []
        for data in batch:
            data = self._toAmperes(data)
            result.append(self._normalize_by_current(data))
        return result

    def decompose(self, data_batch: list, level = None, phase_name = "Ia", wavename = "bior3.1", verbose = 0):
        '''
        Returns:
            batch: list of pd.DataFrames, columns are decomposition coefficients, rows are units.
        '''
        wavelet = pywt.Wavelet(wavename)
        result = []
        
        iterable_object = tqdm.tqdm(data_batch, total = len(data_batch)) if verbose >= 1 else data_batch
        for unit in iterable_object:

            unit = self._toAmperes(unit[phase_name])
            unit = self._normalize_by_current(unit)

            dec = pd.DataFrame(pywt.wavedec(unit, wavelet, level=level)).dropna(axis=1).T #drop all NAs and transpose to ordinar shape
            dec = pd.melt(dec)["value"].to_numpy() #flatten to 1dim numpy array
            result.append(dec)
        return np.array(result)

class TempRWE_Encoder(Decomposer):
    '''
    Temporal Relative Wavelet Energy Encoder. Acts like ordinar RWE-encoder, but keeps the 
    greatest approximation level non-summed, only squared. It allows embedding to be informative w.r.t.
    moment of time fault occured. All detail coefficients are squared and summed-up level-wise as for ordinar RWE.
    '''
    def __init__(self, maxcurrent=500, wavename = 'rbio1.3', maxlevel = None, verbosity = 0):
        super().__init__(maxcurrent)
        self.wavename = wavename
        self.maxlevel = maxlevel
        self.verbosity = verbosity

    @staticmethod
    def _get_total_energy(levels_list):
        """
        Computes Total Energy
        """

        lvl_flat = np.concatenate(levels_list, axis=0)

        return (lvl_flat**2).sum(axis=0)
    
    def decompose(self, data_batch, phase_name = None):
        '''
        Builds stacked Temporal RWE Embeddings. Embedding concatenates approx level and RWE of detail levels: 
        
        - A^2 / E_tot           - vector of size (Na,); Na - size of A
        - sum_i(D[k][i]) / E_tot     - vector of size (M-1,); M - number of decomposition levels;
        k = 0...M-2
        '''
        wavename = self.wavename
        verbose = self.verbosity
        level = self.maxlevel

        wavelet = pywt.Wavelet(wavename)
        
        result = []
        # print(f'databatch shape: {data_batch.shape}')
        # print(pywt.dwt_max_level(data_len=data_batch.shape[-1], filter_len=wavelet.dec_len))

        iterable_object = tqdm.tqdm(data_batch, total = len(data_batch)) if verbose >= 1 else data_batch
        for unit in iterable_object:

            if phase_name:
                unit = self._toAmperes(unit[phase_name])
                unit = self._normalize_by_current(unit)

            dec_unit = pywt.wavedec(unit, level=level, wavelet=wavelet)

            TOTAL_ENERGY = self._get_total_energy(dec_unit)

            approx_level = dec_unit[0]**2
            # print(approx_level.shape)
            # print()
            detail_levels = dec_unit[1:]
            # for dlevel in detail_levels:
                
            #     print(dlevel.shape)
            RelEns = np.array([(dlevel**2).sum() for dlevel in detail_levels])

            

            embedding = np.concatenate([approx_level / TOTAL_ENERGY, RelEns / TOTAL_ENERGY])
            result.append(embedding)
        # print(approx_level.shape)
        return np.array(result)

class EmptyEncoder(Decomposer):

    def __init__(self, maxcurrent=500):
        super().__init__(maxcurrent)

    def decompose(self, data_batch, level=None, phase_name="Ia", wavename="bior3.1", verbose=0):
        return data_batch

class WindowSamplerTorch:
    def __init__(self, data, labels, wsize=80, stride=4, idx_start=0, padding_mode="same", smp_per_period = 80):
        assert data.dim() == 2, "Data must have shape (num_units, unit_len)"
        assert labels.dim() == 2, "Labels must have shape (num_units, unit_len)"

        self.data = data
        self.labels = labels
        self.wsize = wsize
        self.stride = stride
        self.idx_start = idx_start
        self.padding_mode = padding_mode

        self.smp = smp_per_period

        self._pad()

    def _pad(self):
        
        if self.padding_mode == "same":
            pad_values = self.data[:, :self.wsize]      #takes first wsize elems
            pad_labels = self.labels[:, :self.wsize]    #takes first wsize elems
        elif self.padding_mode == "zeros":
            pad_values = torch.zeros((self.data.shape[0], self.wsize), dtype=self.data.dtype)
            pad_labels = torch.zeros((self.labels.shape[0], self.wsize), dtype=self.labels.dtype)
        else:
            raise ValueError("Unsupported padding mode")

        n = (self.wsize - 1) // self.smp + 1
        
        pad_values = pad_values.repeat(1, n)[:, -self.wsize+1:] #takes last wsize-1 elems
        pad_labels = pad_labels.repeat(1, n)[:, -self.wsize+1:] #takes last wsize-1 elems

        self.data = torch.cat([pad_values, self.data], dim=1)
        self.labels = torch.cat([pad_labels, self.labels], dim=1)

    def get_all_windows(self):
        
        windows = self.data[:, self.idx_start:].unfold(1, self.wsize, self.stride)  # (num_units, num_windows, wsize)
        labels = self.labels[:, self.idx_start:].unfold(1, self.wsize, self.stride)[:, :, -1]
        
        return windows, labels

    def __getitem__(self, idx):
        """ Возвращает окна для конкретного индекса вдоль unit_len """
        return self.get_all_windows()[0][:, idx, :], self.get_all_windows()[1][:, idx]

    def __len__(self):
        """ Количество окон, доступных для семплирования """
        return (self.data.shape[1] - self.idx_start - self.wsize) // self.stride + 1
    
class WindowSampler():

    def __init__(self, 
                 data, 
                 labels, 
                 wsize = 80, 
                 stride=4,
                 start_idx=0,
                 padding_mode = 'same'
                 ):
        
        assert len(data.shape) == 2, 'Wrong axes in `data`. Exactly 2 axis is allowed: (unit, values)'
        assert len(labels.shape) == 2, 'Wrong axes in `labels`. Exactly 2 axis is allowed: (unit, values)'

        self.origshape = data.shape
        self.data = data
        self.labels = labels
        self.wsize = wsize
        self.stride = stride

        self.idx_start = start_idx
        self.padmode = padding_mode

        self._pad()

        pass
    
    def _pad(self):
        
        if self.padmode == "same":
            sinpad = self.data[:, 0:80]     #shape (m, 80)
            binpad = self.labels[:, 0:80]   #shape (m, 80)
        elif self.padmode == "zeros":
            sinpad = np.zeros((self.data.shape[0], 80))
            binpad = np.zeros((self.data.shape[0], 80))
        else:
            raise ValueError
        
        n = (self.wsize - 1) // 80 + 1
        sinpad = np.tile(sinpad, (1, n))
        binpad = np.tile(binpad, (1, n))

        sinpad = sinpad[:, -self.wsize:]  # shape (m, self.wsize)
        binpad = binpad[:, -self.wsize:]  # shape (m, self.wsize)

        self.data = np.concatenate([sinpad, self.data], axis=1)
        self.labels = np.concatenate([binpad, self.labels], axis=1)
        pass
    
    
    def _windows_by_pos(self, matrix2d, wpos):

        return matrix2d[:, wpos+1 : wpos+1+self.wsize] #we are in coordinates of padded data. 
                                                    
        #taking window [0, wsize] is the same as take window of original data [wpos - wsize, wpos]

    def _get_windows_by_pos(self, wpos):
        '''
        Takes `m` windows from `data` which have right edges at index `wpos` along time-axis in the original `data` array
        Returns matrix of size `(m, wsize)`, `m` - dataset size, `wsize` - window size
        '''

        return self._windows_by_pos(self.data, wpos=wpos)
    
    def _get_binlabels_by_pos(self, wpos):
        '''
        Takes `m` windows from `labels` which have right edges at index `wpos` along time-axis in the original `labels` array
        Returns matrix of size `(m, 1)`, `m` - dataset size
        '''

        return self._windows_by_pos(self.labels, wpos=wpos)
    
    # def __iter__(self, window_index):

    #     wdatas = self._get_windows_by_pos(window_index) # (m, wsize)
    #     wlabels = self._get_binlabels_by_pos(window_index) #(m, 1)
    #     yield wdatas, wlabels

    def __iter__(self):
        """ Делаем класс итерируемым """
        self.current_index = self.idx_start
        return self
    
    def __next__(self):
        """ Returns the batch with next windows"""
        if self.current_index + self.wsize > self.data.shape[1]:
            raise StopIteration

        batch_data, batch_label = self[self.current_index // self.stride] #call of __getitem__
        self.current_index += self.stride  # forwards window
        return batch_data, batch_label

    
    def __getitem__(self, index):
        """ Позволяет делать `sampler[45]` и получать окно с меткой """
        if index < 0 or index >= len(self):
            raise IndexError(f"Index {index} out of range")

        wpos = self.idx_start + index * self.stride
        return self._get_windows_by_pos(wpos), self._get_binlabels_by_pos(wpos)

    def __len__(self):
        """ Возвращает количество возможных окон """
        return (self.origshape[1] - self.idx_start) // self.stride

class DatasetWindowed(Dataset):
    def __init__(self, data, labels, wsize=80, stride=4, start_idx=0, padding_mode='same'):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.uint8)
        self.wsize = wsize
        self.stride = stride
        self.start_idx = start_idx
        self.padding_mode = padding_mode

        
        self.sampler = WindowSamplerTorch(
            self.data, self.labels, wsize=self.wsize, stride=self.stride, idx_start=self.start_idx, padding_mode=self.padding_mode
        )
        
        self.all_windows, self.all_labels = self.sampler.get_all_windows()  # (num_units, num_windows, wsize), (num_units, num_windows)

    def __len__(self):
        return len(self.all_windows)  

    def __getitem__(self, index):
        """
        - data: (num_windows, wsize)
        - labels: (num_windows,)
        """
        unit_windows = self.all_windows[index]  # (num_windows, wsize)
        unit_labels = self.all_labels[index]  # (num_windows,)

        return unit_windows.to(dtype=torch.float32), unit_labels.to(dtype=torch.int8)
    
class WaveletEmbeddingDataset(Dataset):
    def __init__(self, data, labels, Encoder, wsize=80, stride=4, start_idx=0, padding_mode='same' ):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.uint8)
        self.wsize = wsize
        self.stride = stride
        self.start_idx = start_idx
        self.padding_mode = padding_mode
        self.Encoder = Encoder

        
        self.sampler = WindowSamplerTorch(
            self.data, self.labels, wsize=self.wsize, stride=self.stride, idx_start=self.start_idx, padding_mode=self.padding_mode
        )
        
        self.all_windows, self.all_labels = self.sampler.get_all_windows()  # (num_units, num_windows, wsize), (num_units, num_windows)


        # for window in self.all_windows:  

    def __len__(self):
        return len(self.all_labels)  

    def __getitem__(self, index):
        """
        - data: (num_windows, wsize)
        - labels: (num_windows,)
        """
        unit_windows = self.all_windows[index]  # (num_windows, wsize)
        unit_labels = self.all_labels[index]  # (num_windows,)

        embeddings = torch.tensor(self.Encoder.decompose(unit_windows), dtype = torch.float32)
        

        return embeddings.to(dtype=torch.float32), unit_labels.to(dtype=torch.int8)
    
class PrecomputedWaveletEmbeddingDataset(WaveletEmbeddingDataset):

    def __init__(self, data, labels, Encoder, wsize=80, stride=4, start_idx=0, padding_mode='same'):
        super().__init__(data, labels, Encoder, wsize, stride, start_idx, padding_mode)
        
        # self.all_labels = self.all_labels.unsqueeze(-1)
        print(f"Labels shape: {self.all_labels.shape}\n")
        window = self.all_windows[0]
        generic_emb = self.Encoder.decompose(window)
        self.all_embeddings = torch.zeros(
            (
                len(self), 
                generic_emb.shape[-2], 
                generic_emb.shape[-1]
                ), 
            dtype=torch.float32)
        print(f"Allocated Tensor for All Embeddings shape: {self.all_embeddings.shape}\n")
        self._precompute()

        del self.all_windows
    
    def _precompute(self):

        for index in tqdm.tqdm(range(len(self)), total=len(self), desc="Precomputing Embeddings"):

            unit_windows = self.all_windows[index]  # (num_windows, wsize)
            embeddings = torch.tensor(
                self.Encoder.decompose(unit_windows), 
                dtype = torch.float32) #(num_windows, emb_dim)
            self.all_embeddings[index, :, :] = embeddings

    def __getitem__(self, index):
        
        return self.all_embeddings[index], self.all_labels[index]
    
    # def __getitems__(self, indices: List[int]):
    #     return self.all_embeddings[indices], self.all_labels[indices]

class SubBatcher:
    def __init__(self, subseq_len=10):
        self.subseq_len = subseq_len
    
    def _rebatch(self, data_batch, labels_batch):
        """
        data_batch      [batch_size, seq_len, features] -> data_subbatch    [num_subseqs, batch_size, subseq_len, features]
        labels_batch    [batch_size, seq_len]           -> labels_subbatch  [num_subseqs, batch_size, subseq_len]
        """
        batch_size, seq_len, feature_dim = data_batch.shape
        num_subseqs = seq_len // self.subseq_len        #num homo seqs
        
        if num_subseqs * self.subseq_len != seq_len:
    
            seq_len = num_subseqs * self.subseq_len
            data_batch = data_batch[:, :seq_len, :]     #drops inhomo seqs
            labels_batch = labels_batch[:, :seq_len]    #drops inhomo seqs
        
        data_subbatch = data_batch.view(batch_size, num_subseqs, self.subseq_len, feature_dim).permute(1, 0, 2, 3)
        labels_subbatch = labels_batch.view(batch_size, num_subseqs, self.subseq_len).permute(1, 0, 2)
        
        return data_subbatch, labels_subbatch
    
    def __call__(self, data_batch, labels_batch):
        return self._rebatch(data_batch, labels_batch)
    

class SubseqPWEDataset(PrecomputedWaveletEmbeddingDataset):

    def __init__(self, data, labels, Encoder, wsize=80, stride=4, start_idx=0, padding_mode='same', subseq_length=50):
        super().__init__(data, labels, Encoder, wsize, stride, start_idx, padding_mode)
        self.subseq_length = subseq_length
        self.make_subseqs()
    
    def make_subseqs(self):
        subseqs = []
        label_seqs = []

        for unit_idx in range(len(self.all_embeddings)):
            embedding_seq = self.all_embeddings[unit_idx]   # (seq_len, emb_dim)
            label_seq = self.all_labels[unit_idx]           # (seq_len,)

            seq_len = embedding_seq.shape[0]
            trimmed_len = (seq_len // self.subseq_length) * self.subseq_length

            embedding_seq = embedding_seq[:trimmed_len]
            label_seq = label_seq[:trimmed_len]

            subseqs.append(embedding_seq.reshape(-1, self.subseq_length, embedding_seq.shape[-1]))      # (num_subseqs, subseq_length, emb_dim)
            label_seqs.append(label_seq.reshape(-1, self.subseq_length))                                # (num_subseqs, subseq_length)

        self.subseqs = torch.cat(subseqs, dim=0)
        self.subseq_labels = torch.cat(label_seqs, dim=0)

        print(f"Subsequences created with shape: {self.subseqs.shape}")
        print(f"Subsequence labels created with shape: {self.subseq_labels.shape}")
    
    def __getitem__(self, index):
        subseq = self.subseqs[index]                # (subseq_length, emb_dim)
        subseq_label = self.subseq_labels[index]    # (subseq_length,)
        return subseq, subseq_label
    
def custom_metric_torch_batch(y_true: torch.Tensor, y_pred: torch.Tensor, verbose=0) -> torch.Tensor:

    tp = torch.sum((y_pred == 1) & (y_true == 1), dim=-1)
    tn = torch.sum((y_pred == 0) & (y_true == 0), dim=-1)
    fp = torch.sum((y_pred == 1) & (y_true == 0), dim=-1)
    fn = torch.sum((y_pred == 0) & (y_true == 1), dim=-1)

    if verbose:
        print(f'TP: {tp}')
        print(f'TN: {tn}')
        print(f'FP: {fp}')
        print(f'FN: {fn}')
    
    denominator = tp + fp + fn
    metric = torch.where(denominator > 0, tp / denominator, torch.zeros_like(denominator))
    metric = torch.where(tn == y_true.shape[-1], torch.ones_like(tn), metric)
    return metric  # (batch_size,)

def double_metric(y_true: torch.Tensor, y_pred: torch.Tensor, verbose=0):

    faulty = y_true.sum(dim=-1) > 0
    non_faulty = y_true.sum(dim=-1) == 0

    metric_faulty = custom_metric_torch_batch(y_true=y_true[faulty], y_pred=y_pred[faulty], verbose=verbose)
    metric_non_faulty = custom_metric_torch_batch(y_true=y_true[non_faulty], y_pred=y_pred[non_faulty], verbose=verbose)
    
    return metric_faulty, metric_non_faulty

def get_model_preds_probas_targets(model, loader, proba_threshold = 0.5, device="cpu", old = True):
    model.eval()
    

    multipreds = []
    multiprobas = []
    targets = []
    
    with torch.no_grad():
        itr_obj = tqdm.tqdm(loader, total=len(loader), desc=' Evaluation')
        for batch_data, batch_target in itr_obj:
            batch_data, batch_target = batch_data.to(device), batch_target.to(device)
            
            out_logits = model(batch_data)
            out_logits = out_logits[0]
            
            
            probas = torch.sigmoid(out_logits)
            preds_binary = probas >= proba_threshold

            multipreds.append(preds_binary)
            multiprobas.append(probas)
            targets.append(batch_target)


    multipreds = torch.cat(multipreds, dim=0)
    multiprobas = torch.cat(multiprobas, dim=0)
    targets = torch.cat(targets, dim=0)

    return multipreds, multiprobas, targets