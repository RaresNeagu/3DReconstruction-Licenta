import os

DATASET_ROOT = "C:\licenta\Reconstruction3D\dataset"
SHAPENET_ROOT = os.path.join(DATASET_ROOT, "ShapeNetP2M")

ELLIPSOID_PATH = os.path.join(DATASET_ROOT, "info_ellipsoid.dat")

PRETRAINED_VGG16 = os.path.join("C:\licenta\Reconstruction3D\dataset\data\pretrained", "vgg16.pth")
PRETRAINED_ResNet50 = os.path.join("C:\licenta\Reconstruction3D\dataset\data\pretrained", "resnet50.pth")

IMG_NORM_MEAN = [0.485, 0.456, 0.406]
IMG_NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224