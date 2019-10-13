from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

import logging
# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


class VGG:

  @staticmethod
  def mean_center(arr, undo="False"):
    """Mean center input tensors for VGG 

    Args:
      arr: np.array
      undo: booleand, undo mean-centering operation, e.g. for display
    Returns:
      np.array of image data with values mean-centered for use with VGG
    """
    if isinstance(arr, (list, tuple)): 
      return [VGG.mean_center(o) for o in arr]

    if isinstance( arr, np.ndarray):
      rank = len(arr.shape)
      if rank in (2,3,4):
        rgb_mean = np.asarray( [103.939,116.779,123.68] ) # r,g,b values range(0,255)
        if arr.dtype != 'uint8':
          # assume array has been normalized
          rgb_mean = np.divide( rgb_mean, 255., dtype=np.float32 )
        if undo=="False": 
          return np.subtract(arr.copy(), rgb_mean, dtype=np.float32)
        else:
          return np.add(arr.copy(), rgb_mean, dtype=np.float32) 
      else:
        log.warning( "np.array rank invalid, should be rank in (2,3,4)")
        return arr
        

  @staticmethod
  def rgb_reverse(arr):
    """reverse RGB channels to BGR

    Args:
      arr: np.array of shape (b,h,w,3) or (h,w,3)
    Returns:
      np.array of rank=3 with RGB channels reversed
    """
    if isinstance(arr, (list, tuple)): 
      return [VGG.rgb_reverse(o) for o in arr]

    x = arr.copy()
    assert arr.shape[-1] == 3, ("Input image(s) must be of shape=(b,h,w,3) or (h,w,3)")
    rank = len(arr.shape)
    if rank in (3,4) and arr.shape[-1] == 3:
      return x[..., ::-1]
      

  @staticmethod
  def apply_preprocessing(images, mean_center=True, bgr_ordering=True, undo=False):
    """apply standard VGG pre-processing
    Args:
      images: array or list of rgb images
      mean_center: boolean
      bgr_ordering: boolean
      undo: boolean, default False. reverse preprocessing for display

    Returns:
      batch or list of images as nparrays
    """
    if isinstance(images, (list, tuple)): 
      return [VGG.apply_preprocessing(image, mean_center, bgr_ordering, undo) for image in images]

    if undo and bgr_ordering:
      # return to rgb ordering BEFORE mean_center()
      images = VGG.rgb_reverse(images)
    if mean_center:
      images = VGG.mean_center(images, undo=undo)

    if undo == False and bgr_ordering:
      images = VGG.rgb_reverse(images)
    return images



from skimage.transform import resize

class Image:

  @staticmethod
  def smaller(arr, scale=2.):
    """shrink image in np.array form

    shrink image in nparray form by scale, default is shrink by factor of 2
    Args:
      arr: image as np.array
      scale: default 2.
    Returns:
      image as np.array
    """
    *dim,_ = arr.shape
    target =  tuple( np.array(dim)//scale )
    return resize( arr, target ).astype('float32')