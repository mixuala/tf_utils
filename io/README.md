These modules' names end on 'ing' so the three main public methods can stay as
simple verbs ('load', 'save', and 'show') without creating namespace conflicts
if you do need to import the module or some of the lower level methods.

> `tf_utils.io` cloned from: https://github.com/tensorflow/lucid commit=3f72cb29453665f1ad3258b75dcca0c211fc20fa

```
git clone  https://github.com/tensorflow/lucid.git lucid-commit-3f72cb2
cd ./lucid-commit-3f72cb2
git checkout 3f72cb29453665f1ad3258b75dcca0c211fc20fa --depth 1
```

added support for use with tensorflow 2.0 environments
```
<!-- import tensorflow as tf -->
if tf.__version__.startswith("2."):
  from tensorflow.io import gfile
  # monkey_patch alias `gfile.Open()`
  setattr(gfile, "Open", gfile.GFile) 
else:
  from tensorflow import gfile
```

