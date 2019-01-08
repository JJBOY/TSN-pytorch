import torchvision
import random
from PIL import Image, ImageOps
import numpy as np
import numbers
import torch


class GroupRandomCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size

        out_images = list()

        assert w - th >= 0 and h - th >= 0, \
            'the to be croped image should bot small than crop size'
        x1 = random.randint(0, w - th)
        y1 = random.randint(0, h - th)

        for img in img_group:
            out_images.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_images


class GroupScale(object):
    # 放缩到的size只保证最短的边是size，宽高等比缩放
    def __init__(self, size, interpolation=Image.BILINEAR):
        self.worker = torchvision.transforms.Resize(size, interpolation)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupCenterCrop(object):
    # 都从中间切割
    def __init__(self, size):
        self.worker = torchvision.transforms.CenterCrop(size)

    def __call__(slef, img_group):
        return [slef.worker(img) for img in img_group]


class GroupRandomHorizontalFlip(object):
    # Randomly horizontally flips the given PIL.Image with a probability of 0.5
    def __init__(self, is_flow=False):
        self.is_flow = is_flow

    def __call__(self, img_group):
        v = random.random()
        if v < 0.5:
            ret = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in img_group]
            if self.is_flow:
                for i in range(0, len(ret), 2):
                    ret[i] = ImageOps.invert(ret[i])  # 255-pixel_value
            return ret
        else:
            return img_group


class GroupMultiScaleCrop(object):
    # 宽高比抖动
    # 返回一个随机切割的部分并resize到指定大小
    def __init__(self, input_size, scales=None, max_distort=1, fix_crop=True, more_fix_crop=True):
        self.scales = scales if scales is not None else [1, 0.875, 0.75]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = input_size if not isinstance(input_size, int) else [input_size] * 2
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop((offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation)
                         for img in img_group]

        return ret_img_group

    def _sample_crop_size(self, im_size):
        # 宽高比抖动，计算crop的宽高和偏移
        # fix_crop和more_fix_crop是用来选择偏移的
        # 从None->fix_crop->more_fix_crop，截取的位置越来越随机

        image_w, image_h = im_size[0], im_size[1]
        # find a crop size
        base_size = min(image_w, image_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, image_w - crop_pair[0])
            h_offset = random.randint(0, image_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(image_w, image_h, crop_pair[0], crop_pair[1])
        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(self, image_w, image_h, crop_w, crop_h):
        offsets = self.fill_fix_offset(self.more_fix_crop, image_w, image_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(more_fix_crop, image_w, image_h, crop_w, crop_h):
        w_step = (image_w - crop_w) // 4
        h_step = (image_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))
        ret.append((4 * w_step, 0))
        ret.append((0, 4 * h_step))
        ret.append((4 * w_step, 4 * h_step))
        ret.append((2 * w_step, 2 * h_step))

        if more_fix_crop:
            ret.append((0, 2 * h_step))
            ret.append((4 * w_step, 2 * h_step))
            ret.append((2 * w_step, 4 * h_step))
            ret.append((2 * w_step, 0 * h_step))

            ret.append((1 * w_step, 1 * h_step))
            ret.append((3 * w_step, 1 * h_step))
            ret.append((1 * w_step, 3 * h_step))
            ret.append((3 * w_step, 3 * h_step))
        return ret


class Stack(object):
    # 经过这一步，一组图片已经被拼成了一幅图片的格式(w,h,c)
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        # return (w,h,c)
        if img_group[0].mode == 'L':  # [seg*len],(w,h)
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB':  # [seg*len],(w,h,3)
            if self.roll:
                return np.concatenate([np.array(x)[:, :, ::-1] for x in img_group], axis=2)
            else:
                return np.concatenate(img_group, axis=2)


class ToTorchFormatTensor(object):
    """ Converts a PIL.Image (RGB) or numpy.ndarray (H x W x C) in the range [0, 255]
    to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0] """

    def __init__(self, div=True):
        self.div = div

    def __call__(self, pic):
        if isinstance(pic, np.ndarray):  # (HWC)
            img = torch.from_numpy(pic).permute(2, 0, 1).contiguous()
        else:  # (WHC)
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
            img = img.view(pic.size[1], pic.size[0], len(pic.mode))  # (HWC)
            img = img.transpose(0, 1).transpose(0, 2).contiguous()  # (WHC)->(CHW)
        return img.float().div(255) if self.div else img.float()


class IdentityTransform(object):
    def __call__(self, data):
        return data


class GroupNormalize(object):
    # 注意：这里的GroupNormalization不是插入在网络层里面的只是在输入之前对数据进行归一化
    def __init__(self, mean, std):
        self.mean = mean  # [C]
        self.std = std

    def __call__(self, input):
        # input shape:(CHW)
        rep_mean = self.mean * (input.size()[0] // len(self.mean))
        rep_std = self.std * (input.size()[0] // len(self.std))

        for t, m, s in zip(input, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return input


if __name__ == '__main__':
    pass
