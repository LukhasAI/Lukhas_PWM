import numpy as np
from perception.sensor_fusion import fuse_lidar_and_camera


def test_sensor_fusion_basic():
    lidar = np.array([[0, 0, 0], [1, 1, 1]])
    cam = np.array([[255, 0, 0], [0, 255, 0]])
    fused = fuse_lidar_and_camera(lidar, cam)
    assert fused.shape == (2, 6)
    assert fused[0, 3] == 255
