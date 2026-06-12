

import random

import torch
import torchvision
from torchvision.transforms import v2 as T


class Crop(torch.nn.Module):

    def __init__(self, **kwargs):
        super().__init__()
        crop_kwargs = dict(kwargs)
        self.fill = crop_kwargs.pop("fill", -1)

        self.random_resized_crop = T.RandomResizedCrop(
            size=crop_kwargs.pop("size", (112, 112)),
            scale=crop_kwargs.pop("scale", (.2, 1.)),
            ratio=crop_kwargs.pop("ratio", (0.75, 1.3333333333333333)),
            **crop_kwargs,
        )

    def forward(self, img):
        new_img = torch.full_like(img, self.fill)
        i, j, h, w = self.random_resized_crop.get_params(
            img,
            self.random_resized_crop.scale,
            self.random_resized_crop.ratio)
        cropped = T.functional.crop(img, i, j, h, w)
        new_img[:,i:i+h,j:j+w] = cropped
        return new_img


class LowRes(torch.nn.Module):

    def __init__(self, min_res=None, max_res=None, base_size=112, interpolation_types=None):
        super().__init__()
        self.min_res = base_size * .2 if min_res is None else min_res
        self.max_res = base_size * 1. if max_res is None else max_res

        self._interpolation_types = _resolve_interpolation_types(interpolation_types) if interpolation_types is not None else [
            torchvision.transforms.InterpolationMode.BILINEAR, 
            torchvision.transforms.InterpolationMode.NEAREST, 
            torchvision.transforms.InterpolationMode.NEAREST_EXACT, 
            torchvision.transforms.InterpolationMode.BILINEAR,
            torchvision.transforms.InterpolationMode.BICUBIC
            ]
        
        self.resize_transform = T.functional.resize

    def forward(self, img):
        res_ = int(random.uniform(self.min_res, self.max_res))
        inter_type = random.choice(self._interpolation_types)

        original_size = img.shape[1]

        img = self.resize_transform(img, [res_, res_], interpolation=inter_type)
        return self.resize_transform(img, [original_size, original_size], interpolation=inter_type)


def _resolve_interpolation_types(interpolation_types):
    resolved = []
    for interpolation_type in interpolation_types:
        if isinstance(interpolation_type, str):
            resolved.append(getattr(torchvision.transforms.InterpolationMode, interpolation_type.upper()))
        else:
            resolved.append(interpolation_type)
    return resolved

class Augmentor(torch.nn.Module):

    def __init__(
        self,
        enable,
        probabilities=None,
        color_jitter_kwargs=None,
        low_res_kwargs=None,
        crop_kwargs=None,
    ):
        super().__init__()

        self.enable = enable

        default_probabilities = {
            "color_jitter": .2,
            "low_res": .2,
            "crop": .2,
        }
        probabilities = {**default_probabilities, **(probabilities or {})}

        color_jitter_kwargs = {
            "brightness": .5,
            "contrast": .5,
            "saturation": .5,
            "hue": .0,
            **(color_jitter_kwargs or {}),
        }
        low_res_kwargs = low_res_kwargs or {}
        crop_kwargs = crop_kwargs or {}

        self.color_jitter = T.Compose(
            [T.Lambda(lambda img: (img + 1.) / 2.),
             T.ColorJitter(
                **color_jitter_kwargs
              ),
            T.Normalize([.5, .5, .5], [.5, .5, .5])]
        ) 
        self.low_res_transform = LowRes(**low_res_kwargs)
        self.crop_transform = Crop(**crop_kwargs)
        self.all_transforms = (
            self.color_jitter,
            self.low_res_transform,
            self.crop_transform,
        )
        self.probabilities = (
            probabilities["color_jitter"],
            probabilities["low_res"],
            probabilities["crop"],
        )


    def forward(self, img):

        if not self.enable:
            return img

        for trans, prob in zip(self.all_transforms, self.probabilities):
            if random.random() < prob:
                img = trans(img)

        return img
