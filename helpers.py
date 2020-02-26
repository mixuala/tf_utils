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
      undo: boolean, undo mean-centering operation and clipped to domain, e.g. for display
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

    if tf.is_tensor( arr ):
      return VGG.tf_mean_center(arr)

    assert isinstance( arr, np.ndarray), "expecting ndarray"
    rank = len(arr.shape)
    if rank in (2,3,4):
      # r,g,b values for arr.dtype=='uint8', range(0,255)
      # rgb_mean = np.asarray( [123.68, 116.779, 103.939] ) 
      rgb_mean = VGG.rgb_mean.copy()
      is_normalized = False if arr.dtype == 'uint8' else np.max(arr<=1.0)
      if is_normalized:
        # normalize rgb_mean
        rgb_mean = np.divide( rgb_mean, 255., dtype=np.float32 )

      if undo == False: 
        # assume arr is already clipped to correct bounds
        # print( "mean_center RGB/APPLY, is_normalized=",is_normalized, rgb_mean )
        return np.subtract(arr.copy(), rgb_mean, dtype=np.float32)
      else:
        # print( "mean_center RGB/REMOVE, is_normalized=",is_normalized, rgb_mean, )
        # print(" .  ",tf.reduce_mean(vgg_transfer_image, axis=[1,2], keepdims=True))
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

    assert tf.is_tensor( arr ), "expecting a tensor"

    rank = len(arr.shape)
    if rank in (2,3,4):
      # r,g,b values for arr.dtype=='uint8', range(0,255)
      # rgb_mean = np.asarray( [123.68, 116.779, 103.939] ) 
      rgb_mean = tf.convert_to_tensor(VGG.rgb_mean, dtype=tf.float32)
      is_normalized = False if arr.dtype == tf.uint8 else tf.less_equal(tf.math.reduce_max(arr),1.0) is not None
      if is_normalized:
        # normalize rgb_mean
        rgb_mean = tf.divide( rgb_mean, tf.constant(255., dtype=tf.float32) )

      if undo == False: 
        # assume arr is already clipped to correct bounds
        # tf.print( "mean_center RGB/APPLY, is_normalized=",is_normalized )
        return tf.subtract(arr, rgb_mean)
      else:
        # tf.print( "tf_mean_center RGB/REMOVE, is_normalized=",is_normalized, rgb_mean, )
        # tf.print(" .  ",tf.reduce_mean(vgg_transfer_image, axis=[1,2], keepdims=True))
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

    # print("rgb_reverse(): arr=", type(arr))
    x = tf.identity(arr) if tf.is_tensor(arr) else arr.copy()
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
      # print("bgr => RGB", images.shape)
      # return to rgb ordering BEFORE mean_center()
      images = VGG.rgb_reverse(images)

    if mean_center: # expecting images to be in RGB ordering
      # print("mean_center, undo=",undo, images.shape)
      images = VGG.mean_center(images, undo=undo)

    if undo == False and bgr_ordering:
      # print("RGB => bgr, shape=", images.shape)
      images = VGG.rgb_reverse(images)
    return images



class ImgStacker():
  """log a series of learned images with the same shape

  typically a series of samples from one epoch in a row
  with each row from successive epochs
  """
  _shape=None
  h_items=[]
  v_items=[]

  def _check_shape(self, image):
    shape = tf.squeeze(image).shape
    if self._shape is None:
      assert len(shape)==3 # shape=(h,w,c)
      self._shape = shape
    else:
      h,w,c = shape
      assert h==self._shape[0] and w==self._shape[1] and c==self._shape[2], "expecting shape={}, got: {}".format(self._shape, shape)

  def clear(self):
    self._shape=None
    self.h_items=[]
    self.v_items=[]

  def shape(self):
    return (len(self.v_items), len(self.h_items))

  def hstack(self, image=None, limit=10, smaller=True):
    """stack a horizonal row of images into 1 ndarray, most recent first
    
      Args:
        image, shape=(h,w,c), if None, returns current hstack
      Returns: shape=(h,n*w,c), n<=limit
    """
    if image is not None:
      if smaller:
        image = Image.smaller(np.squeeze(image))
      self.h_items.insert(0, image) # most recent on the left
      self.h_items = self.h_items[:limit]
    return np.concatenate( np.asarray(self.h_items), axis=1 )  # shape=(h,n*w,c)

  def vstack(self, row=None, limit=10):
    """stack a vertical row of images into 1 ndarray, most recent last
    
      Args:
        row, shape=(h,w,c), if None, returns current hstack
      Returns: shape=(n*h,w,c), n<=limit
    """
    if row is not None:
      if len(self.v_items)>0:
        assert row.shape==self.v_items[0].shape, "expecting row of shape={}".format(self.v_items[0].shape)
      self.v_items.insert(0, row) # most recent on the top
      self.v_items = self.v_items[:limit]
    if len(self.v_items)>0:
      return np.concatenate( np.asarray(self.v_items), axis=0 )  # shape=(n*h,w,c)
    



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
    if hasattr(arr, 'numpy') and callable(arr.numpy):
      arr = arr.numpy()
    h,w,c = arr.shape
    dim = (np.array([h,w])//scale)
    return skimage_resize( arr, dim , anti_aliasing=True).astype('float32')


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

    target_shape = shape[:2]
    _resize_cmd = tf.image.resize if tf.is_tensor(arr) else skimage_resize

    # # see also: tf.image.resize( input, target_shape )
    # x = tf.image.resize(x, target_shape)

    if (resize_method is None): 
      return _resize_cmd( arr, shape[:2], anti_aliasing=True )

    if (resize_method is 'contained'):
      isContained = np.max(np.array(arr.shape[:2]) - np.array(shape[:2]))
      if (isContained <=0 ):
        return arr
      ratio = np.max(np.array(arr.shape) / np.array(shape))
      target_shape = (arr.shape[:2] / ratio).astype(int)
      # print('target=', target_shape, arr.shape, shape)
      return _resize_cmd( arr, target_shape, anti_aliasing=True )

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
        # add content and style image
        ordered += np.squeeze(np.delete(results[0].copy(), transfer_index, axis=0)).tolist()
      # insert style_transfer image in the middle
      ordered.insert(insert_position, np.squeeze(results[i][transfer_index]) )  
      
    hstacked = np.concatenate( np.asarray(ordered), axis=1 )
    # print("hstacked images, count=", results.shape[0]+2 ,hstacked.shape)
    return hstacked

  @staticmethod
  def upload_imgur(dataURL, pick=None):
    """upload base64 dataURL to imgur and get static url

    see: https://apidocs.imgur.com/?version=latest#c85c9dfc-7487-4de2-9ecd-66f727cf3139

    usage: 
      dataurl="data:image/PNG;base64,[...]"
      check = helpers.Image.upload_imgur(imgdata)
      print(check['link'])

    NOTE: max dataURL size is MAX_ARG_STRLEN = 131072 bytes

    Args:
      dataURL: string
      pick: pick keys to return, see API. default=['id', 'link', 'deletehash']

    Returns: dict={id:, link:, deletehash:}
    """
    from html import escape
    from json import loads
    CLIENT_ID = "a098034b70f2f30"
    MAX_ARG_STRLEN = 131072
    default_pick = ['id', 'link', 'deletehash']
    pick = pick if pick is not None else default_pick

    assert dataURL.startswith("data:image"), "ERROR: expecting an image dataURL" 
    dataURL = dataURL.split(',',1).pop() # strip base64 prefix

    # ## NOTE: additional post params are not working
    # extras = ""
    # if title is not None:
    #   extras += "&title={}".format( escape(title) )
    # if desc is not None:
    #   extras += "&description={}".format( escape(desc) )
    # print("extras", extras)
    # payload += extras

    assert len(payload) < MAX_ARG_STRLEN, "ERROR: MAX_ARG_STRLEN exceeeded"
    import requests
    headers = {
        'Authorization': 'Client-ID $CLIENT_ID',
    }
    resp = requests.post('https://api.imgur.com/3/image', headers=headers, data={'image':escape(dataURL)})
    # resp = !curl --location --request POST "https://api.imgur.com/3/image" \
    #   --header "Authorization: Client-ID $CLIENT_ID " \
    #   --form "$payload"
    resp = loads(resp[0])
    assert resp['success'], "Imgur API error: {}".format(resp)
    data = resp['data']
    # print(data)
    result = { k:v for k,v in data.items() if k in pick and v is not None }
    return result