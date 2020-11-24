# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import json
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset

def get_dataset(name):
    return {'JSONAudioDataset': JSONAudioDataset,
            'JSONAudioSpeakerDataset': JSONAudioSpeakerDataset}[name]

def get_partition(utts, i_split, n_split):
    '''Getting i-th partition of dictionary.

    Used to parallelize inference.

    Args:
        utts (dict): dictionary with utterances as keys
        i_split (int): index of desired partition
        n_split (int): number of partitions to split into

    Returns:
        utts_part (dict): i-th partition of utts
    '''
    n_per_split = len(utts) // n_split + 1
    idx_from = i_split * n_per_split
    idx_to = (i_split + 1) * n_per_split
    keys_split = list(utts.keys())[idx_from:idx_to]
    utts_part = {u:utts[u] for u in keys_split}
    return utts_part

class JSONAudioDataset(Dataset):
    def __init__(self, path, transform = None, i_split = 0, n_split = 1):
        '''Audio dataset described by JSON file.

        The JSON file should contain a dictionary in form:
        {
         '<utt-key1>' : {'path': [<path-to-wav-file1>]},
         '<utt-key2>' : {'path': [<path-to-wav_file2>]},
         ...
        }

        Args:
            path (str): Path to JSON file describing the dataset.
            transform (function): function to apply on the audio data
            i_split (int): index of partition of the dataset to load
            n_split (int): number of partitions to split the dataset into
            
        '''
        self.transform = transform if transform is not None else lambda x: x
        with open(path) as f:
            utts_all = json.load(f)
        self.utts = get_partition(utts_all, i_split, n_split)

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = list(self.utts.keys())[idx] if isinstance(idx, int) else idx
        s, fs = sf.read(self.utts[utt]['path'][0])

        # for multi-channel signal, use 1st channel
        if s.ndim > 1:
            s = s[:, 0]
        
        s = self.transform(s.astype(np.float32))
        return s

class JSONAudioSpeakerDataset(Dataset):
    def __init__(self, path, transform = None, i_split = 0, n_split = 1):
        '''Audio dataset described by JSON file, including speaker information.

        The JSON file should contain a dictionary in form:
        {
         '<utt-key1>' : {'path': [<path-to-wav-file1>],
                         'speaker': '<speaker-id1>'},
         '<utt-key2>' : {'path': [<path-to-wav_file2>],
                         'speaker': '<speaker-id2>'},
         ...
        }

        Args:
            path (str): Path to JSON file describing the dataset.
            transform (function): function to apply on the audio data
            i_split (int): index of partition of the dataset to load
            n_split (int): number of partitions to split the dataset into
            
        '''
        self.transform = transform if transform is not None else lambda x: x
        with open(path) as f:
            utts_all = json.load(f)
        self.utts = get_partition(utts_all, i_split, n_split)

        speakers = list(set([utts_all[u]['speaker'] for u in utts_all]))
        self.spk2id = {spk: idx for idx, spk in enumerate(speakers)}

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = list(self.utts.keys())[idx] if isinstance(idx, int) else idx
        s, fs = sf.read(self.utts[utt]['path'][0])

        # for multi-channel signal, use 1st channel
        if s.ndim > 1:
            s = s[:, 0]
        
        s = self.transform(s.astype(np.float32))
        spkid = self.spk2id[self.utts[utt]['speaker']]
      
        return s, spkid

class JSONAudioMultichannelDataset(Dataset):
    def __init__(self, path, transform = None, 
                       i_split = 0, n_split = 1, 
                       channels = None):
        '''Multichannel audio dataset described by JSON file.

        The JSON file should contain a dictionary in form:
        {
         '<utt-key1>' : {'path': [<path-to-wav-file1-channel1>,
                                  <path-to-wav-file1-channel2>,
                                  ...],
                         'speaker': '<speaker-id1>'},
         '<utt-key2>' : {'path': [<path-to-wav-file2-channel1>,
                                  <path-to-wav-file2-channel2>,
                                  ...],
                         'speaker': '<speaker-id2>'},
         ...
        }
        There also may be multiple channels in one wav-file. In that case,
        we expect shape n_samples x n_channels after loading the file with
        soundfile (this is the case of spatialized WSJ0-2MIX).

        Args:
            path (str): Path to JSON file describing the dataset.
            transform (function): function to apply on the audio data
            i_split (int): index of partition of the dataset to load
            n_split (int): number of partitions to split the dataset into
            channels (list): indices of channels to load, None for loading all 
        '''
        self.transform = transform if transform is not None else lambda x: x
        with open(path) as f:
            utts_all = json.load(f)
        self.utts = get_partition(utts_all, i_split, n_split)
        self.channels = channels

    def __len__(self):
        return len(self.utts)

    def __getitem__(self, idx):
        utt = list(self.utts.keys())[idx] if isinstance(idx, int) else idx

        # the channels of the signal can be either as individual files
        # or all in one file
        s_all = []
        for f in self.utts[utt]['path']:
            s, fs = sf.read(f)
            if s.ndim > 1:
                s_all.extend([self.transform(s1) for s1 in s.T])
            else:
                s_all.append(self.transform(s))

        minlen = np.min([s.shape[0] for s in s_all])
        s_all = [s[:minlen] for s in s_all] # just in case of channel failure
        s = np.stack(s_all)

        if self.channels is not None:
            s = s[self.channels]
      
        return s
