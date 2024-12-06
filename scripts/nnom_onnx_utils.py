'''
    Copyright (c) 2024-
    Jiamingyy
    jiamingyy@outlook.com

    SPDX-License-Identifier: Apache-2.0

    Change Logs:
    Date           Author       Notes
    2024-12-05     Jiamingyy   The first version


    This file provides:
        -> some tools function used in nnom
'''
import numpy as np
import scipy


def find_dec_bits_max_min(data, bit_width=8, maximum_bit=32):
    """
    A ragular non-saturated shift-based quantization mathod. Using max/min values
    :param data: array-like data
    :param bit_width: quantization bit width, default is 8
    :param maximum_bit: maximum decimal bit. Incase sometime bias is too small lead to very large size dec bit
    :return:
    """
    max_val = abs(data.max()) - abs(data.max() / pow(2, bit_width))  # allow very small saturation.
    min_val = abs(data.min()) - abs(data.min() / pow(2, bit_width))
    int_bits = int(np.ceil(np.log2(max(max_val, min_val))))
    dec_bits = (bit_width - 1) - int_bits
    return min(dec_bits, maximum_bit)


def find_dec_bits_max_min_axis(data, axis=-1, bit_width=8, maximum_bit=32):
    """
    A ragular non-saturated shift-based quantization mathod. Using max/min values
    :param data:
    :param axis:
    :param bit_width:
    :return: a tuple of optimal dec bits for each axis
    """
    dec_bits = []
    # if(len(data.shape) < np.abs(axis)): # for depthwise with axis = -2 while len(shape) =1
    #     size = data.shape[0]
    #     axis = 0 #
    # else:
    #     size = data.shape[axis]
    for i in np.arange(0, data.shape[axis]):
        d = np.take(data, indices=i, axis=axis)
        max_val = abs(d.max()) - abs(d.max() / pow(2, bit_width))  # allow very small saturation.
        min_val = abs(d.min()) - abs(d.min() / pow(2, bit_width))
        int_bit = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
        dec_bit = (bit_width - 1) - int_bit
        dec_bits.append(min(dec_bit, maximum_bit))
    return dec_bits


def find_dec_bits_kld(data, bit_width=8, scan_times=4, maximum_bit=16):
    """
    # saturation shift, using KLD method (Kullback-Leibler divergence)
    # Ref: http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
    :param data: The data for looking for quantization
    :param bit_width: the bitwidth of the data
    :param scan_times: the times to try the best kld (normally the second is the best.)
    :return: dec bit width for this data
    """
    # do a regular non-saturated quantization
    max_val = data.max()
    min_val = data.min()
    abs_max = max(abs(max_val), abs(min_val))
    int_bits = int(np.ceil(np.log2(max(abs(max_val), abs(min_val)))))
    dec_bits = (bit_width - 1) - int_bits

    # now looking for the best quantization using KLD method
    small_var = 1e-5
    bins = np.arange(-abs_max, abs_max, abs_max / 2048 * 2)
    q_bins = np.arange(-abs_max, abs_max, abs_max / 256 * 2)
    flat_hist = np.histogram(data.flatten(), bins=bins)[0]
    kl_loss = []
    kl_shifts = []
    for shift in range(scan_times):
        t = 2 ** (dec_bits + shift)  # 2-based threshold
        act = np.round(data.flatten() * t)
        act = act / t
        act = np.clip(act, -128 / t, 127 / t)
        act = np.histogram(act, bins=q_bins)[0]
        act_hist = np.zeros(2047)
        chunk = int(2048 / 256)
        for i in range(int(255)):
            none_zero = np.count_nonzero(flat_hist[i * chunk:(i + 1) * chunk])
            if none_zero == 0:
                continue
            for j in range(chunk):
                act_hist[i * chunk + j] = act[i] / none_zero if flat_hist[i * chunk + j] != 0 else 0
        flat_hist[flat_hist == 0] = small_var
        act_hist[act_hist == 0] = small_var
        kl = scipy.stats.entropy(flat_hist, act_hist)
        kl_loss.append(kl)
        kl_shifts.append(dec_bits + shift)

    # now get the least loss from the scaned kld shift
    dec_bits = kl_shifts[np.argmin(kl_loss)]  # set the dec_bit to the KLD results
    return min(dec_bits, maximum_bit)


# convert to [-127,127] fixed point
def quantize_data(data, dec_bits, axis=-1, per_axis=False, bitwith=8):
    """
    Convert the data to the given bitwidth and dec bits using Fixed point Quantization method.
    :param data: the data to be quantized
    :param dec_bits: the dec bits for quantization
    :param axis: the axis to be quantized
    :param per_axis: if the dec_bits is per axis
    :param bitwith: the bitwidth of the data, default is 8
    """
    if per_axis:
        out = []
        for i in np.arange(0, data.shape[axis]):
            d = np.take(data, indices=i, axis=axis)
            d = np.round(d * 2 ** dec_bits[i])
            d = np.clip(d, -2 ** (bitwith - 1), 2 ** (bitwith - 1) - 1)
            d = np.expand_dims(d, axis=axis)
            out.append(d)
        out = np.concatenate(out, axis=axis)
        return out
    else:
        return np.clip(np.round(data * 2 ** dec_bits), -2 ** (bitwith - 1), 2 ** (bitwith - 1) - 1)
