# kitti_depthmap
Module for converting raw lidar in kitti to depthmaps in C++/Python.

## Why I wrote this
Surprisingly, I wasn't able to find any decent/simple/readable code online to figure out how to generate registered depthmaps from the kitti pointcloud data. And those that did, gave garbage results (when actually visualized as a pointcloud), or were written with for loops in python which were way too slow for a data pipeline. This codebase handles parallax errors that come from a large transformation between a sparse 3d sensor and an rgb image, and also performs resampling in spherical coordinate space, leading to more accurate results.
. 
### To Run
```
mkdir build;
cd build;
cmake ..;
make
python3 testing.py # Change the path in the file to point to the kitti data
```

### Large Image
Accounting for parallax  distortion etc.

![Large Image](https://github.com/soulslicer/kitti_depthmap/blob/master/large_img.png?raw=true)

### Small Image
Works well for small images too

![Small Image](https://github.com/soulslicer/kitti_depthmap/blob/master/small_img.png?raw=true)

### Sampling Modes (Lower)
Selectively simulate a higher or lower resolution lidar in spherical space

![Uniform Image](https://github.com/soulslicer/kitti_depthmap/blob/master/uniform_img.png?raw=true)

### Upsampling (Higher)
Selectively simulate a higher or lower resolution lidar in spherical space

![Before Image](https://github.com/soulslicer/kitti_depthmap/blob/master/before_upsample.png?raw=true)

![After Image](https://github.com/soulslicer/kitti_depthmap/blob/master/after_upsample.png?raw=true)




