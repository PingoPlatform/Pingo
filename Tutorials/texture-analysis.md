---
description: 'Author: Peter Feldens'
---

# Texture Analysis

## Texture Analysis

We use the Virtual machine provided in the "Links to data" chapter. 

For texture analysis, we will keep things simpler and just classify different subsets of a side scan sonar mosaic using the inbuild functions of Python’s scikit-image module, which offers several functions for image classification.

For a “real world” application, the classification we are about to do would need to be applied to complete images, and the results to be stored in a dataframe similar to the ARA analysis \(see chapter on introduction to ARA analysis\). The texture parameters could then be fed, e.g., in the automated machine learning algorithms \(see chapter machine learning techniques\).

We switch to the folder

cd /home/ecomap/ecomap\_summerschool/texture

and load a fresh instance of jupyter qtconsole. In this folder, there are three example images displaying a nadir artefact, a stone area and common coarse sand seafloor. Each image is 10x10 m in size. Well will calculate several texture parameters for these images.

```python
%pylab inline
import numpy as np
import os
os.chdir('/home/ecomap/ecomap_summerschool/texture')
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops
from skimage import io
import sys
sys.path.append('/home/ecomap/ecomap_summerschool/python_functions')
import ecomap_summerschool_functions as eco

nadir = io.imread('nadir.png', as_gray = True)
stone = io.imread('stone.png', as_gray = True)
sediment = io.imread('sediment.png', as_gray = True)
```

We can plot the images using

```python
plt.imshow(nadir)
plt.imshow(stone)
plt.imshow(sediment)
```

![](../.gitbook/assets/image%20%285%29.png)

![](../.gitbook/assets/image%20%288%29.png)

![](../.gitbook/assets/image%20%289%29.png)

The differences in spatial pixel arrangement are obvious. Let us calculate the texture parameters. 

```python
texture_parameters = [] file_list = [nadir, stone, sediment] 
labels = ['nadir', 'stone', 'sediment'] 
for i, img in enumerate(file_list): # The parameters given to greycomatrix are img, distance, angle, greylevels 
    img = eco.reduce_greylevels(img,31) 
    glcm = greycomatrix(img, [1], [0, 1.57], 32, symmetric=True, normed=True) 
    homo = greycoprops(glcm, 'homogeneity')[0, 0] 
    corr = greycoprops(glcm, 'correlation')[0, 0] 
    contr = greycoprops(glcm, 'contrast')[0, 0] 
    energ = greycoprops(glcm, 'energy')[0, 0] 
    texture_parameters.append([labels[i], homo, corr, contr, energ]) 
    Texture_Results = pd.DataFrame(texture_parameters, columns = ['Label', 'Homogeneity', 'Correlation', 'Contrast', 'Energy']) 
    Texture_Results.groupby('Label').mean()
```

```text
          Homogeneity  Correlation   Contrast    Energy
Label                                                  
nadir        0.271841     0.577876  68.992105  0.116639
sediment     0.178274     0.529739  69.034211  0.058252
stone        0.217238     0.639262  47.655263  0.065947
```

We can observe that different texture parameters are able to differentiate between the different images.

