import utils_lib as pyutils
import numpy as np
import pykitti
import cv2
import time

def torgb(img):
    # Load RGB
    rgb_cv = np.asarray(img).copy()
    rgb_cv = cv2.cvtColor(rgb_cv, cv2.COLOR_RGB2BGR)
    rgb_cv = rgb_cv.astype(np.float32)/255
    return rgb_cv

def plot_depth(dmap_raw_np, rgb_cv, name="win"):
    rgb_cv = rgb_cv.copy()

    dmax = np.max(dmap_raw_np)
    dmin = np.min(dmap_raw_np)
    for r in range(0, dmap_raw_np.shape[0], 1):
        for c in range(0, dmap_raw_np.shape[1], 1):
            depth = dmap_raw_np[r, c]
            if depth > 0.1:
                dcol = depth/20
                rgb_cv[r, c, :] = [1-dcol, dcol, 0]
                #cv2.circle(rgb_cv, (c, r), 1, [1-dcol, dcol, 0], -1)

    cv2.namedWindow(name)
    cv2.moveWindow(name, 2500, 50)
    cv2.imshow(name, rgb_cv)
    cv2.waitKey(15)

# Parameters
basedir = "kitti"
date = "2011_09_26"
drive = "0005"
# date = "2011_10_03"
# drive = "0047"

# KITTI Load
p_data = pykitti.raw(basedir, date, drive, frames= range(5, 10))
indx = 0

# IMU to camera #
intr_raw = None
raw_img_size = None  # [1226, 370]
mode = "left"
if mode == "left":
    M_imu2cam = p_data.calib.T_cam2_imu
    M_velo2cam = p_data.calib.T_cam2_velo
    intr_raw = p_data.calib.K_cam2
    raw_img_size = p_data.get_cam2(0).size
    img = p_data.get_cam2(indx)
elif mode == "right":
    M_imu2cam = p_data.calib.T_cam3_imu
    M_velo2cam = p_data.calib.T_cam3_velo
    intr_raw = p_data.calib.K_cam3
    raw_img_size = p_data.get_cam3(0).size
    img = p_data.get_cam3(indx)

# Load Velodyne Data
velodata = p_data.get_velo(indx)  # [N x 4] [We could clean up the low intensity ones here!]
velodata[:, 3] = 1.

# Large Image Depthmap
large_img_size = (768/1,256/1)
uchange = float(large_img_size[0])/float(raw_img_size[0])
vchange = float(large_img_size[1])/float(raw_img_size[1])
intr_large = intr_raw.copy()
intr_large[0,:] *= uchange
intr_large[1,:] *= vchange
intr_large_append = np.append(intr_large, np.array([[0, 0, 0]]).T, axis=1)
large_img = cv2.resize(torgb(img), large_img_size, interpolation=cv2.INTER_LINEAR)
large_params = {"filtering": 2, "upsample": 0}
dmap_large = pyutils.generate_depth(velodata, intr_large_append, M_velo2cam, large_img_size[0], large_img_size[1], large_params)
plot_depth(dmap_large, large_img, "large_img")

# Small Image Depthmap
small_img_size = (768/4,256/4)
uchange = float(small_img_size[0])/float(raw_img_size[0])
vchange = float(small_img_size[1])/float(raw_img_size[1])
intr_small = intr_raw.copy()
intr_small[0,:] *= uchange
intr_small[1,:] *= vchange
intr_small_append = np.append(intr_small, np.array([[0, 0, 0]]).T, axis=1)
small_img = cv2.resize(torgb(img), small_img_size, interpolation=cv2.INTER_LINEAR)
small_params = {"filtering": 0, "upsample": 0}
dmap_small = pyutils.generate_depth(velodata, intr_small_append, M_velo2cam, small_img_size[0], small_img_size[1], small_params)
plot_depth(dmap_small, small_img, "small_img")

# Upsampled
upsampled_img_size = (768/2,256/2)
uchange = float(upsampled_img_size[0])/float(raw_img_size[0])
vchange = float(upsampled_img_size[1])/float(raw_img_size[1])
intr_upsampled = intr_raw.copy()
intr_upsampled[0,:] *= uchange
intr_upsampled[1,:] *= vchange
intr_upsampled_append = np.append(intr_upsampled, np.array([[0, 0, 0]]).T, axis=1)
upsampled_img = cv2.resize(torgb(img), upsampled_img_size, interpolation=cv2.INTER_LINEAR)
upsampled_params = {"filtering": 1, "upsample": 4}
dmap_upsampled = pyutils.generate_depth(velodata, intr_upsampled_append, M_velo2cam, upsampled_img_size[0], upsampled_img_size[1], upsampled_params)
plot_depth(dmap_upsampled, upsampled_img, "upsampled_img")

# Uniform Sampling
uniform_img_size = (768/1,256/1)
uchange = float(uniform_img_size[0])/float(raw_img_size[0])
vchange = float(uniform_img_size[1])/float(raw_img_size[1])
intr_uniform = intr_raw.copy()
intr_uniform[0,:] *= uchange
intr_uniform[1,:] *= vchange
intr_uniform_append = np.append(intr_uniform, np.array([[0, 0, 0]]).T, axis=1)
uniform_img = cv2.resize(torgb(img), uniform_img_size, interpolation=cv2.INTER_LINEAR)
uniform_params = {"filtering": 2, "upsample": 1,
                  "total_vbeams": 64, "vbeam_fov": 0.4,
                  "total_hbeams": 750, "hbeam_fov": 0.4}
dmap_uniform = pyutils.generate_depth(velodata, intr_uniform_append, M_velo2cam, uniform_img_size[0], uniform_img_size[1], uniform_params)
plot_depth(dmap_uniform, uniform_img, "uniform_img")

# Interpolation Pytorch Test

cv2.waitKey(0)
