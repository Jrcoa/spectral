import torch
import torchvision.transforms as transforms
# I want to create torch transforms to rotate images by a random degree
# and to flip images horizontally and vertically

class RandomRotate(transforms.RandomRotation):
    def __init__(self, degrees=90, prob=0.95, interpolation=transforms.InterpolationMode.NEAREST, expand=False, center=None):
        super(RandomRotate, self).__init__(degrees, interpolation, expand, center)
        self.prob = prob

    def __call__(self, x, label):
        # angle = self.get_params(self.degrees)
        angle = torch.rand(1).item() * self.get_params(self.degrees)
        angle = -angle if torch.rand(1).item() < 0.5 else angle
        
        if torch.rand(1).item() < self.prob:
            return transforms.functional.rotate(x, angle, self.interpolation, self.expand, self.center), transforms.functional.rotate(label, angle, self.interpolation, self.expand, self.center)
        return x, label
    
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __init__(self, p=0.5):
        super(RandomHorizontalFlip, self).__init__(p)
    
    def __call__(self, x, label):
        if torch.rand(1).item() < self.p:
            return transforms.functional.hflip(x), transforms.functional.hflip(label)
        return x, label

class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __init__(self, p=0.5):
        super(RandomVerticalFlip, self).__init__(p)
    
    def __call__(self, x, label):
        if torch.rand(1).item() < self.p:
            return transforms.functional.vflip(x), transforms.functional.vflip(label)
        return x, label
    
class Compose(transforms.Compose):
    def __call__(self, x, label):
        for t in self.transforms:
            x, label = t(x, label)
        return x, label