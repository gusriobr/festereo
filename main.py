import os
import sys
from collections import namedtuple
from random import randint

import torch
from PIL import Image
from torchvision import transforms

from pl_template import FEStereo


def load_pair(base_folder, img_name, downsample=False):
    l_im = Image.open("{}/left/{}".format(base_folder, img_name))
    r_im = Image.open("{}/right/{}".format(base_folder, img_name))

    if downsample:
        w, h = l_im.size
        th, tw = 256, 512

        x1 = randint(0, w - tw)
        y1 = randint(0, h - th)

        l_im = l_im.crop((x1, y1, x1 + tw, y1 + th))
        r_im = r_im.crop((x1, y1, x1 + tw, y1 + th))

    return l_im, r_im  # , l_disp


# -------------------------------------------------------------------
#  Get image transform
# -------------------------------------------------------------------
def get_transform():
    # Prepare transform
    mean = [0.0, 0.0, 0.0]
    std = [1.0, 1.0, 1.0]
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean, std)])
    return transform


def predict(base_folder, img_name):
    global output
    # left, right = load_pair("/media/data/datasets/depth/Sampler/Driving/RGB_cleanpass", img_name)
    # left, right = load_pair("/media/data/datasets/depth/Sampler/FlyingThings3D/RGB_cleanpass", img_name)
    left, right = load_pair(base_folder, img_name, downsample=True)

    trf = get_transform()
    left = trf(left)
    right = trf(right)
    # left = transforms.ToTensor()(left).unsqueeze_(0)
    # right = transforms.ToTensor()(right).unsqueeze_(0)
    # left = trf.transforms(left)
    # right = trf.transforms(right)
    output = model(left.unsqueeze(0), right.unsqueeze(0))
    img = transforms.ToPILImage()(output.squeeze(0))
    img.save("/tmp/{}_disp.png".format(img_name))


if __name__ == '__main__':
    use_gpu = False
    if len(sys.argv) > 1:
        use_gpu = True

    if not use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        check_point = torch.load("model/sceneflow_ckpt_epoch_19.ckpt", map_location=torch.device('cpu'))
        print("Using CPU")
    else:
        check_point = torch.load("model/sceneflow_ckpt_epoch_19.ckpt")
        print("Using GPU")

    # print(states_dict)
    DATASET_BASE = "/media/data/datasets/depth/Sampler"

    check_point["hparams"]["datasets_path"] = DATASET_BASE
    hparams = namedtuple("Object", " ".join(list(check_point["hparams"].keys())))(*check_point["hparams"].values())
    model = FEStereo(hparams)

    model.load_state_dict(check_point["state_dict"])
    for fd in ['sceneflow/monkaa/RGB_cleanpass/', 'FlyingThings3D/RGB_cleanpass/', 'Driving/RGB_cleanpass/']:
        base_folder = os.path.join(DATASET_BASE, fd)
        lst_imgs = os.listdir(os.path.join(base_folder,'left'))
        for img_name in lst_imgs:
            predict(base_folder, img_name)
