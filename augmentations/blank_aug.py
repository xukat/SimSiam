# import torchvision.transforms as T
# try:
#     from torchvision.transforms import GaussianBlur
# except ImportError:
#     from .gaussian_blur import GaussianBlur
#     T.GaussianBlur = GaussianBlur
# 
# imagenet_mean_std = [[0.485, 0.456, 0.406],[0.229, 0.224, 0.225]]

class BlankTransform():
    def __init__(self):
        self.transform = None
        self.transform_list = []
    def __call__(self, x):
        if self.transform is None:
            print("WARNING: no augmentation")
            return x, x

        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2

