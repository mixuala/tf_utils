These modules' names end on 'ing' so the three main public methods can stay as
simple verbs ('load', 'save', and 'show') without creating namespace conflicts
if you do need to import the module or some of the lower level methods.

> `tf_utils.io` cloned from: https://github.com/tensorflow/lucid v0.3.9-alpha

added support for use with tensorflow 2.0 environments
```
import tensorflow as tf
if tf.__version__.startswith("2."):
  gfile = tf.io.gfile
  # monkey_patch alias `gfile.Open()`
  setattr(gfile, "Open", gfile.GFile) 
else:
  from tensorflow import gfile
```