# image-processing

Contains a set of Python tools related to image processing.


Requirements
------------

* NumPy
* SciPy
* pandas (optional)
* rasterio
* fiona
* shapely
* Rtree (optional)
* libspatialindex (optional)
* idl-functions


Segmentation analysis example
-----------------------------

```python
import numpy
from scipy import ndimage
from image_processing.segmentation import Segments

dims = (1000, 1000)
class_data = numpy.random.randint(0, 10001, dims)
nlabels = ndimage.label(class_data > 5000, output=class_data)
seg = Segments(class_data)

data = numpy.random.ranf(dims)

# pandas DataFrame example
stats = seg.basic_statistics(data, dataframe=True)
print stats.head(10)
print stats["Mean"].head(10)

# compound NumPy array example
stats_c = seg.basic_statistics(data)
print stats_c[0:10]
print stats_c["Mean"][0:10]

print "Number of segments: {}".format(seg.n_segments)

# data associated with a given segment
seg_data = seg.data(data, segment_id=seg.ids[0])
print seg_data

# remove segments containing < 10 pixels
seg.sieve(10)
print "Number of segments: {}".format(seg.n_segments)

# remove segments containing < 30 pixels
seg.sieve(30)
print "Number of segments: {}".format(seg.n_segments)
```


Contrast enhancement example
----------------------------

```python
import numpy
import matplotlib.pyplot as plt
from image_processing import contrast

dims = (1000, 1000)
data = numpy.random.ranf(dims)

lp = contrast.linear_percent(data, percent=3)
lg = contrast.log(data)
sq = contrast.square_root(data)
eq = contrast.equalisation(data)

fig = plt.figure()
fig.add_subplot(221)
plt.title('Linear 3%')
plt.imshow(lp)
fig.add_subplot(222)
plt.title('Log')
plt.imshow(lg)
fig.add_subplot(223)
plt.title('Square Root')
plt.imshow(sq)
fig.add_subplot(224)
plt.title('Equalisation')
plt.imshow(eq)
plt.show()
```
