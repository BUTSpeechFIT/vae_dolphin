# Copyright (c) 2020 Brno University of Technology
# Copyright (c) 2020 Nippon Telegraph and Telephone corporation (NTT).
# All rights reserved
# By Katerina Zmolikova, November 2020. 

import torch

# https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
def pad_tensor(vec, pad, dim):
    '''
    Args:
        vec: tensor to pad
        pad: the size to pad to
        dim: dimension to pad

    return:
        a new tensor padded to 'pad' in dimension 'dim'
    '''
    pad_size = list(vec.shape)
    pad_size[dim] = pad - vec.shape[dim]
    return torch.cat([vec, torch.zeros(*pad_size)], dim=dim)

def pad_collate_more(batch, dim):
    '''
    For the case that batch element contains multiple items.
    (E.g. audio data and speaker-ids)
    '''
    all = []
    all_nontensor = []
    for i in range(len(batch[0])):
        batch_i = [b[i] for b in batch]

        if isinstance(batch_i[0], int):
            batch_i_pad = torch.tensor(batch_i, dtype = torch.long)
            all_nontensor.append(batch_i_pad)
        else:
            batch_i_pad, lengths = pad_collate_one(batch_i, dim)
            all.append(batch_i_pad)

    return (*all, *all_nontensor, lengths)

def pad_collate_one(batch, dim):
    # find longest sequence
    lengths = [x.shape[dim] for x in batch]
    max_len = max(lengths)
    # pad according to max_len
    batch = [pad_tensor(torch.Tensor(x), pad=max_len, dim=dim)
                for x in batch]
    # stack all
    xs = torch.stack(batch, dim=0)

    lengths = torch.LongTensor(lengths)

    return xs, lengths

# https://discuss.pytorch.org/t/dataloader-for-various-length-of-data/6418/8
class PadCollate:
    """
    a variant of collate_fn that pads according to the longest sequence in
    a batch of sequences
    """

    def __init__(self, dim=0):
        """
        args:
            dim - the dimension to be padded (dimension of time in sequences)
        """
        self.dim = dim

    def __call__(self, batch):
        """
        args:
            batch - list of tensors

        return:
            xs - a tensor of all examples in 'batch' after padding
        """
        if isinstance(batch[0], tuple):
            return pad_collate_more(batch, self.dim)
        else:
            return pad_collate_one(batch, self.dim)
