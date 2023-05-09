import os

# dataset root
DATASET_ROOT = "C:\licenta\Reconstruction3D\dataset"
SHAPENET_ROOT = os.path.join(DATASET_ROOT, "ShapeNetP2M")

# ellipsoid path
ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "info_ellipsoid.dat")

# Mean and standard deviation for normalizing input image
IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224