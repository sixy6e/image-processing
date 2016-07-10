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
from image_processing.segmentation import Segments

dims = (1000, 1000)
class_data = numpy.random.randint(0, 256, dims)
seg = Segments(class_data)

data = numpy.random.randf(dims)

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

# remove segments containing > 30 pixels
seg.sieve(30)
print "Number of segments: {}".format(seg.n_segments)
```
