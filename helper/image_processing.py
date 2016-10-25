#!/usr/bin/env
# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pdb

"""
图像的预处理工作
1. resize
2. 去均值
3. 填充至统一大小
"""






def resize(im, target_size, max_size):
    """
    only resize input image to target size and return scale
    :param im: BGR image input by opencv
    :param target_size: one dimensional size (the short side)
    :param max_size: one dimensional max size (the long side)
    :return:
    """
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)

    im = cv2.resize(im, None, None, fx=float(target_size) / float(im_shape[1]), fy=float(target_size) / float(im_shape[0]), interpolation=cv2.INTER_LINEAR)
    # im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR)
    return im, im_scale



def transform(im, pixel_means):
    """
    transform into mxnet tensor
    substract pixel size and transform to correct format
    :param im: [height, width, channel] in BGR
    :param pixel_means: [[[R, G, B pixel means]]]
    :return: [batch, channel, height, width]
    """
    im = im.copy()
    # reverse channel
    # BGR -> RGB
    im[:, :, (0, 1, 2)] = im[:, :, (2, 1, 0)]
    im = im.astype(float)
    im -= pixel_means
    im_tensor = im[np.newaxis, :]
    # put channel first
    channel_swap = (0, 3, 1, 2)
    im_tensor = im_tensor.transpose(channel_swap)
    return im_tensor



def tensor_vstack(tensor_list, pad=0,stack = True):
    """
    vertically stack tensors
    :param tensor_list: list of tensor to be stacked vertically
    :param pad: label to pad with
    :return: tensor with max shape
    """
    ndim = len(tensor_list[0].shape)
    if ndim == 1:
        return np.hstack(tensor_list)
    dimensions = [0]
    for dim in range(1, ndim):
        dimensions.append(max([tensor.shape[dim] for tensor in tensor_list]))
    # pad操作
    # 填充缺失的点
    # 将不同长宽比的图片尺寸统一起来
    # 然后通过vstack将list变成np.array
    for ind, tensor in enumerate(tensor_list):
        pad_shape = [(0, 0)]
        for dim in range(1, ndim):
            pad_shape.append((0, dimensions[dim] - tensor.shape[dim]))
        tensor_list[ind] = np.lib.pad(tensor, pad_shape, 'constant', constant_values=pad)
    if stack==False:
        return tensor_list
    all_tensor = np.vstack(tensor_list)
    return all_tensor