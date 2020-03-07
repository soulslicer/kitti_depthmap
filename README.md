# kitti_depthmap
module for converting raw lidar in kitti to depthmaps in C++/Python

### To Run
```
mkdir build;
cd build;
cmake ..;
make
python3 testing.py
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




