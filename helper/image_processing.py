import cv2
import numpy as np




def tensor_vstack(tensor_list, pad=0):
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
    all_tensor = np.vstack(tensor_list)
    return all_tensor