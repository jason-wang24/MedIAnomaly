import os
from torchvision import transforms

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_PATH = os.path.join(PROJECT_ROOT, 'data')

def get_transform(opt):
    normalize = transforms.Normalize((0.5,), (0.5,)) if opt.model['in_c'] == 1 else \
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    transform = transforms.Compose([transforms.ToTensor(),
                                    normalize])
    return transform


def get_data_path(dataset):
    data_root = BASE_PATH
    if dataset == 'rsna':
        return os.path.join(data_root, "RSNA")
    elif dataset == 'vin':
        return os.path.join(data_root, "VinCXR")
    elif dataset == 'brain':
        return os.path.join(data_root, "BrainTumor")
    elif dataset == 'lag':
        return os.path.join(data_root, "LAG")
    elif dataset == 'brats':
        return os.path.join(data_root, "BraTS2021")
    elif dataset == 'c16':
        return os.path.join(data_root, "Camelyon16")
    elif dataset == 'oct':
        return os.path.join(data_root, "OCT2017")
    elif dataset == 'colon':
        return os.path.join(data_root, "Colon_AD_public")
    elif dataset == 'isic':
        return os.path.join(data_root, "ISIC2018_Task3")
    elif dataset == 'cpchild':
        return os.path.join(data_root, "CP-CHILD", "CP-CHILD-A")
    else:
        raise Exception("Invalid dataset: {}".format(dataset))
