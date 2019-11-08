from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf

import logging
# create logger with module name, e.g. lucid.misc.io.reading
log = logging.getLogger(__name__)


class VGG:

  # these values are constant for VGG
  rgb_mean = np.asarray( [123.68, 116.779, 103.939] ) 


  @staticmethod
  def mean_center(arr, undo=False):
    """Mean center input tensors for VGG 

    apply imagenet zero-mean  to batch
    NOTE: expects RGB ordering, don't forget to reverse to BGR in next step
    Args:
      arr: np.array with RGB ordering
      undo: boolean, undo mean-centering operation, e.g. for display
    Returns:
      np.array of image data with values mean-centered, but NOT explicitly BGR ordered or normalized
    """

    # def normalize_batch(batch):
    #   # normalize using imagenet mean and std
    #   mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    #   std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    #   batch = batch.div_(255.0)
    #   return (batch - mean) / std

    if isinstance(arr, (list, tuple)): 
      return [VGG.mean_center(o) for o in arr]

    if isinstance( arr, np.ndarray):
      rank = len(arr.shape)
      if rank in (2,3,4):
        # r,g,b values for arr.dtype=='uint8', range(0,255)
        # rgb_mean = np.asarray( [103.939,116.779,123.68] )
        rgb_mean = VGG.rgb_mean.copy()
        is_normalized = False if arr.dtype == 'uint8' else np.max(arr<=1.0)
        if is_normalized:
          # normalize rgb_mean
          rgb_mean = np.divide( rgb_mean, 255., dtype=np.float32 )

        if undo == False: 
          # assume arr is already clipped to correct bounds
          return np.subtract(arr.copy(), rgb_mean, dtype=np.float32)
        else:
          undid = np.add(arr.copy(), rgb_mean, dtype=np.float32)
          #  clip to correct bounds
          maxVal = 1. if is_normalized else 255.
          return np.clip(undid, 0.,  maxVal)
      else:
        log.warning( "np.array rank invalid, should be rank in (2,3,4)")
        return arr

  @staticmethod
  def tf_mean_center(arr, undo=False):
    """Mean center input tensors for VGG, for use in tf.graph()

    typically applied to style transfer image which is initialized as tf.random.uniform()

    Args:
      arr: tf.tensor
      undo: boolean, undo mean-centering operation
    Returns:
      tf.tensor of image data with values mean-centered, but NOT explicitly BGR ordered or normalized
    """

    if isinstance( arr, tf.Tensor):
      rank = len(arr.shape)
      if rank in (2,3,4):
        # r,g,b values for arr.dtype=='uint8', range(0,255)
        # rgb_mean = np.asarray( [103.939,116.779,123.68] )
        rgb_mean = tf.convert_to_tensor(VGG2.rgb_mean, dtype=tf.float32)
        is_normalized = False if arr.dtype == tf.uint8 else tf.less_equal(tf.math.reduce_max(arr),1.0) is not None
        if is_normalized:
          # normalize rgb_mean
          rgb_mean = tf.divide( rgb_mean, tf.constant(255., dtype=tf.float32) )

        if undo == False: 
          # assume arr is already clipped to correct bounds
          return tf.subtract(arr, rgb_mean)
        else:
          undid = tf.add(arr, rgb_mean)
          #  clip to correct bounds
          maxVal = 1. if is_normalized else 255.
          return tf.clip_by_value(undid, 0.,  maxVal)
      else:
        log.warning( "tf.Tensor rank invalid, should be rank in (2,3,4)")
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
      batch or list of images as ndarrays with mean_center and BGR ordering, but NOT normalized
    """
    if isinstance(images, (list, tuple)): 
      return [VGG.apply_preprocessing(image, mean_center, bgr_ordering, undo) for image in images]

    if undo and bgr_ordering:
      # return to rgb ordering BEFORE mean_center()
      images = VGG.rgb_reverse(images)

    if mean_center: # expecting images to be in RGB ordering
      images = VGG.mean_center(images, undo=undo)

    if undo == False and bgr_ordering:
      images = VGG.rgb_reverse(images)
    return images



from skimage.transform import resize as skimage_resize

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
    return skimage_resize( arr, target ).astype('float32')

  @staticmethod
  def resize(arr, shape, resize_method='contained'):
    """resize image array using skimage.transform.resize

    use resize_method='contained' to shrink content_image to be contained by the size of shape
    
    Args:
      arr: array shape=(h,w,c)
      shape: target size, shape=(h,w)
      resize_method: 
        'contained' will resize arr so it fits inside shape

    Returns:
      array shape=(h,w,c)
    """

    if isinstance(arr, (list, tuple)): 
      return [Image.resize(o, shape, resize_method) for o in arr]

    rank = len(arr.shape)
    assert rank == 3, ("expecting array of shape=(h,w,c)")

    if ( arr.shape[:2] == shape[:2] ):
      return arr
    if (resize_method is None): 
      return skimage_resize( arr, shape[:2], anti_aliasing=True )
    if (resize_method is 'contained'):
      isContained = np.max(np.array(arr.shape[:2]) - np.array(shape[:2]))
      if (isContained <=0 ):
        return arr
      ratio = np.max(np.array(arr.shape) / np.array(shape))
      target = (arr.shape[:2] / ratio).astype(int)
      # print('target=', target, arr.shape, shape)
      return skimage_resize( arr, target, anti_aliasing=True ) 
      

  def hstack_style_transfer_results( results, transfer_index=0, most_recent=True):
    """hstack a batch of `style_transfer` results

    assumes style_transfer results are an array of shape=(3, h,w,c) e.g. [content_image, transfer_image, style_image] 
    Args:
      results: style_transfer, result or batch of style_transfer results
      transfer_index: index of style_transfer_input in the style_transfer result
      most_recent: style_transfer inserts most_recent result on the left as default
    Returns:
      np.array: 1 hstacked image with the series of style transfer images in the middle, sorted by most recent
    """
    insert_position = 1 if most_recent else -1 
    results = np.asarray(results)
    rank = len(results.shape)
    assert 3 <= rank and rank <=5, ("expecting batch, or batch of batches of visualizations of shape=(h,w,c)")
    if rank == 4: np.expand_dims(results, axis=0) # batch of batches
    assert results.shape[1] ==3, ("expecting 3 images: content, style, and style_transfer") 
    ordered = []
    for i in range(results.shape[0]):
      if len(ordered) ==0: 
        # add first and last
        ordered += np.squeeze(np.delete(results[0].copy(), transfer_index, axis=0)).tolist()
      # insert style_transfer image in the middle
      ordered.insert(insert_position, np.squeeze(results[i][transfer_index]) )  
      
    hstacked = np.concatenate( np.asarray(ordered), axis=1 )
    # print("hstacked images, count=", results.shape[0]+2 ,hstacked.shape)
    return hstacked