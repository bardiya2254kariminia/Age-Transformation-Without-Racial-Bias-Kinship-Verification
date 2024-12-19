from abc import abstractmethod
import torchvision.transforms as transforms
import consts


class TransformsConfig(object):

    @abstractmethod
    def get_transforms(self):
        pass


class EncodeTransforms(TransformsConfig):

    def __init__(self):
        super(EncodeTransforms, self).__init__()

    def get_transforms(self):
        transforms_dict = {
            "transform_gt_train": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_source": None,
            "transform_test": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_inference": transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            ),
            "transform_image_length": transforms.Compose(
                [
                    transforms.Resize((consts.IMAGE_LENGTH, consts.IMAGE_LENGTH)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], inplace=True
                    ),
                ]
            ),
        }
        return transforms_dict
