# Dataset-Augmentation


![alt](https://github.com/JanderHungrige/Dataset-Augmentation/blob/main/1*XF84HS2-X3jbP-cDUBbarQ.png?raw=true)

# Introduction
With the files in this repo you can download creative commons images and adjust them for false images. 
The idea here is that you download images of celebrtites .To avoid jail, you only use creative commons licenced images. However, those are not alwasy the best and lead to false faces. 
With the files here, you can indentifiy false images and remove them. 

The first method would be to compare the images in each folder (one celebrity per folder) and find outliers. Or using the second method to compare all images of a folder to a golden truth from non creative common lincens. 

## Using the Second Method
If you want to compare to the ground trouth, you will have to first download images without the *filter = (commercial, reuse)* and then use the file *2 -Create embeddings database.py* to create so called embeddings. Those Embeddings are then the gold standard. 

**Detailed description can [be found here](https://janwerth.medium.com/8a68be38652?source=friends_link&sk=3f5a8619f10c66c5781aa6b73df0c9eb)**

# Prerequisites
You can download the needed model directly here: 
```
wget ftp://ftp.phytec.de/pub/Software/Linux/Applications/demo-celebrity-face-match-data-1.0.tar.gz
tar -xzf demo-celebrity-face-match-data-1.0.tar.gz
```

* python 3.6+ environment (I recommend [Anaconda](https://www.anaconda.com/) using [virtual environments](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)),
* [icrawler](https://pypi.org/project/icrawler/) , 
* [TensorFlow 2.x](https://pypi.org/project/tensorflow/),
* [tflite_runtime](https://pypi.org/project/tflite/),
* [pandas](https://pypi.org/project/pandas/),
* [numpy](https://pypi.org/project/numpy/),
* [matplotlib](https://pypi.org/project/matplotlib/), 
* [scipy](https://pypi.org/project/scipy/), 
* [opencv-python](https://pypi.org/project/opencv-python/),
* [and the tf.keras-vggface model](https://github.com/JanderHungrige/tf.keras-vggface).

To install the tflite_runtime, download and install [this x86 wheel file](https://github.com/google-coral/pycoral/releases/download/release-frogfish/tflite_runtime-2.5.0-cp36-cp36m-linux_x86_64.whl) and install via pip install path_to_file if the above (ARM) does not work.

# The Files

* **1a-Image-Crawler.py** crawles images with Bing. The filter is set to *commercial, reuse*
* **1b-get_faces_and_crop.py** extracts the face and rescale the image to 224x224
* **1d-proof_images_plotting.py** plots the mean error to see the outliers
* **1e-proof_images_V1.py** determine and delete the outliers based on internal analysis
* **1e-proof_images_V2.py** determine and delete the outliers based on comparison to ground trouth
# License
This project is licensed under the Apache License Version 2.0. See the file *LICENSE* for detailed information.
