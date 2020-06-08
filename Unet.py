import os
import cv2
import random
import math, time

from PyQt5.QtCore import *
from PyQt5.QtWidgets import QMessageBox,QWidget
from PyQt5.QtGui import QImage
import numpy as np
from skimage import io
from skimage import morphology

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tr

from typing import Callable, Iterable, List, Set, Tuple
from model.nets.unet import UNet


class GpuDetector():
    """
    Gpu detector to address the problems related to the gpu.
    """

    def __init__(self):
        self._gpu_number = torch.cuda.device_count()  # if there is np gpu, return 0
        self._gpu_idx_in_use = -1  # which gpu to use
        self._gpu_is_available = False
        for gpu_idx in range(self._gpu_number):
            try:
                self._check_gpu_status(gpu_idx)
            except AssertionError as e:
                pass
            else:
                self._gpu_is_available = True
                self._gpu_idx_in_use = gpu_idx
                break
                # Todo: the multi-gpus inference

    def _set_gpu_device(self, gpu_idx: int) -> Tuple:
        """
        Setting certain gpu advice to use
        """
        try:
            self._check_gpu_status(gpu_idx)
        except AssertionError as e:  # if there is error, raise it to upper call
            raise e
        else:
            self._gpu_is_available = True
            self._gpu_idx_in_use = gpu_idx
            # Todo: the multi-gpus inference

    def _get_current_gpu(self) -> Tuple:
        """
        Get current gpu status and gpu device
        """
        return self._gpu_is_available, self._gpu_idx_in_use

    def _check_gpu_status(self, gpu_idx: int):
        """
        Check gpu status, some errors will return by AssertionError
        """
        assert torch.cuda.is_available(), "This computer has no gpu"
        assert isinstance(gpu_idx, int) and gpu_idx >= 0, "The gpu index must be int and greater than 0"
        assert gpu_idx <= (self._gpu_number - 1), "There is only {} gpus and has no gpu with idx {}".format(
            self._gpu_number, gpu_idx)
        assert torch.cuda.get_device_capability(gpu_idx)[
                   0] >= 2.0, "The device capability of gpu {} with name {} is lower than 2.0".format(gpu_idx,
                                                                                                      torch.cuda.get_device_name(
                                                                                                          gpu_idx))
        # Todo: memory size and useable memory size detection, however this can be handle in RuntimeError in inference


class NetInference():
    """
    Net Inference for 2D image segmentation
    """

    def __init__(self, gpu_detector, input_size: int = 256, overlap_size: int = 32, batch_size: int = 8,
                 aug_type: int = 2, mean: float = 0.9410404628082503, std: float = 0.12481161024777744, model=UNet,
                 pth_address: str = None, use_post_process: bool = True):
        super().__init__()
        # overlap-tile parameters
        self._input_size = input_size
        self._overlap_size = overlap_size

        # batch_size in inference stage can improve the efficiency for overlap-tile strategy，批处理
        self._batch_size = batch_size

        # Test Time augmentation, aug_type = 0: no tta, 1: 4 variants, 2: 8 variants
        self._aug_type = aug_type

        self._mean = mean
        self._std = std
        self._z_score_norm = tr.Compose([
            tr.ToTensor(),
            tr.Normalize(mean=[self._mean],
                         std=[self._std])
        ])
        self._use_post_process = use_post_process

        self._gpu_detector = gpu_detector
        self._model = model()
        gpu_status, gpu_idx_in_use = self._gpu_detector._get_current_gpu()
        if gpu_status:
            self._model = nn.DataParallel(self._model).cuda(gpu_idx_in_use)

        # model parameters setting
        self._pth_address = pth_address
        try:
            self._load_pth()
        except (AssertionError, FileNotFoundError) as e:
            self._load_status = False
        else:
            self._load_status = True

    def _forward_one_image(self, in_ori_img: np.ndarray) -> np.ndarray:
        """
        Forward one image, return segmentation result, some errers will be raised
        """
        assert self._load_status, "Please load parameters correctly"
        assert isinstance(in_ori_img, np.ndarray), "The input image must be numpy type"
        assert in_ori_img.ndim == 2, "The input image should be gray scale"
        assert isinstance(self._input_size,
                          int) and self._input_size % 16 == 0, "The input size must be int and can be divided by 16"
        assert isinstance(self._overlap_size,
                          int) and self._overlap_size > 0, "The overlap size must be int and greater than 0"
        assert self._input_size > (
        2 * self._overlap_size), "The input size {} must be greater than 2 times overlap size {}".format(
            self._input_size, self._overlap_size)
        assert self._aug_type in [0, 1, 2], "The parameter for test time augmentation must be a value in {0,1,2}"
        assert isinstance(self._batch_size,
                          int) and self._batch_size > 0, "The batch num must be int and greater than 0"

        # For overlap tile strategy and tta, the sub imgs are in List Type
        overlap_tile = OverlapTile(crop_size=self._input_size, overlap_size=self._overlap_size)  # 初始化平铺策略的参数
        tta = TestTimeAug(aug_type=self._aug_type)  # 初始化弹性变换方式

        self._model.eval()  # **模型必有
        with torch.no_grad():
            # overlap-tile strategy, crop img into sub-imgs in List and revert after network inference

            in_imgs = overlap_tile.crop(in_ori_img)  # 得到裁剪块大小的图片序列
            in_imgs = tta.aug_input(in_imgs)  # 弹性变换后的图片
            out_imgs = []
            try:
                img_idx = 0
                h, w = in_imgs[0].shape


                while img_idx < len(in_imgs):

                    iter_batch_size = self._batch_size
                    if img_idx + iter_batch_size > len(in_imgs):
                        # if the len(in_imgs) can not be divided by batch_size, the batch size of last itereation should be
                        # len(in_imgs) - img_idx
                        iter_batch_size = len(in_imgs) - img_idx

                    in_tensor = torch.zeros((iter_batch_size, 1, h, w))
                    for b_idx in range(iter_batch_size):
                        in_tensor[b_idx, :, :, :] = self._z_score_norm(in_imgs[img_idx + b_idx])

                    gpu_status, gpu_idx_in_use = self._gpu_detector._get_current_gpu()
                    if gpu_status:
                        in_tensor = in_tensor.cuda(gpu_idx_in_use)
                    out_tensor = self._model.forward(in_tensor)  # b,c,w,h c=2
                    out_tensor = F.softmax(out_tensor, dim=1)  # 将张量的每个元素缩放到（0,1）区间且和为1，dim=1按行
                    if gpu_status:
                        out_tensor = out_tensor.cpu()

                    for b_idx in range(iter_batch_size):
                        # the out data is presented by List again
                        out_imgs.append(out_tensor[b_idx, :, :, :].numpy().transpose((1, 2, 0)))  # 顺序改变从012到120
                    img_idx = img_idx + iter_batch_size

                    torch.cuda.empty_cache()
            except RuntimeError as e:
                raise e  # GPU or CPU memory is out of use
            else:
                out_imgs = tta.merge_out(out_imgs)  # merge different tta results.

                for out_idx in range(len(out_imgs)):
                    # transform type and post process

                    out_imgs[out_idx] = np.argmax(out_imgs[out_idx], axis=2)  # 返回沿轴axis最大值的索引
                    if self._use_post_process:
                        out_imgs[out_idx] = self._post_process(out_imgs[out_idx])

                out_img = overlap_tile.stitch(out_imgs).clip(0, 1).astype(np.uint8)

                # if self._use_post_process:
                #     out_img = self._post_process(out_img)
        return out_img

    def _post_process(self, in_img: np.ndarray) -> np.ndarray:
        """
        Post processing method after network inference.
        """
        # skeletonization of result
        out_img = morphology.skeletonize(1 - in_img, method="lee")  # Lee Skeleton method
        # dialiation with 3 pixels
        out_img = 1 - morphology.dilation(out_img, morphology.square(3))
        # Todo: delete the noise region whose area smaller than threshold
        # Add this depend on the user's decision
        # out_img = morphology.remove_small_objects(out_img, min_size=out_img.shape[0]*out_img.shape[1]*0.001, connectivity=1, in_place=False)
        return out_img

    def _load_pth(self):
        """
        Load model parameters
        """
        assert self._pth_address, "Please set pth address"
        gpu_status, _ = self._gpu_detector._get_current_gpu()
        try:
            if gpu_status:  # load gpu parameters for gpu
                self._model.load_state_dict(torch.load(self._pth_address))
            else:  # load gpu parameters for cpu
                self._model.load_state_dict(
                    {k.replace('module.', ''): v for k, v in torch.load(self._pth_address).items()})
        except FileNotFoundError as e:
            raise e

    def _set_model(self, model):
        self._model = model

    def _set_pth_address(self, pth_address: str):
        self._pth_address = pth_address
        try:
            self._load_pth()
        except (AssertionError, FileNotFoundError) as e:
            self._load_status = False
            raise e
        else:
            self._load_status = True

    def _set_input_size(self, input_size: int):
        assert isinstance(input_size,
                          int) and input_size % 16 == 0, "The input size must be int and can be divided by 16"
        self._input_size = input_size

    def _set_overlap_size(self, overlap_size: int):
        assert isinstance(overlap_size, int) and overlap_size > 0, "The overlap size must be int and greater than 0"
        self._overlap_size = overlap_size

    def _set_batch_size(self, batch_size: int):
        assert isinstance(self._batch_size,
                          int) and self._batch_size > 0, "The batch num must be int and greater than 0"
        self._batch_size = batch_size

    def _set_aug_type(self, aug_type: int):
        assert self._aug_type in [0, 1, 2], "The parameter for test time augmentation must be a value in {0,1,2}"
        self._aug_type = aug_type

    def _set_use_post_process(self, use_post_process: bool):
        assert isinstance(use_post_process, bool), "The use_post_process must be bool"
        self._use_post_process = use_post_process

    def _set_mean_and_std(self, mean: float, std: float):
        assert isinstance(mean, float) and mean > 0, "The mean must be float and greater than 0"
        assert isinstance(std, float) and std > 0, "The std  must be float and greater than 0"
        self._mean = mean
        self._std = std


def load_img(img_path: str) -> np.ndarray:
    """
        Load images or npy files
        :param img_path:  Address for images or npy file
        :return: PIL image or numpy array
    """
    img = io.imread(img_path)
    if np.amax(img) == 255 and len(np.unique(img)) == 2:
        img = img * 1.0 / 255
    return img


class OverlapTile():
    """
    This stategy is implementated from: Ma B, Ban X, Huang H, et al. Deep learning-based image segmentation for al-la alloy microscopic images[J]. Symmetry, 2018, 10(4): 107.
    This is crop function
    For figure 4
    crop_size is the size of blue rectangle, which is equal to input_size
    roi_size is the size of yellow rectangle
    """

    def __init__(self, crop_size: int = 256, overlap_size: int = 32):
        self._crop_size = crop_size
        self._overlap_size = overlap_size  # 裁切的重叠距离
        self._roi_size = crop_size - 2 * overlap_size
        self._in_img_shape = None

    def crop(self, in_img: np.ndarray) -> List:
        """
        Crop in_img to sub-img List, 
        hint: self._roi_size is used in crop and stitch stage, please check the paper careful when read this code
        """
        self._in_img_shape = in_img.shape
        # Pad image before cropping
        in_pad_img = np.pad(in_img, self._overlap_size, mode='symmetric')
        # calculate the number of cropping, which consider the influence of remainder
        h_pad_num = math.ceil(in_pad_img.shape[0] / (self._roi_size))
        w_pad_num = math.ceil(in_pad_img.shape[1] / (self._roi_size))
        #         if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
        #         if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
        in_crop_imgs = []
        for i in range(h_pad_num):
            for j in range(w_pad_num):
                # overlap_crop, it is need to calculate the start of cropping
                start_h = i * self._roi_size;
                end_h = start_h + self._crop_size
                start_w = j * self._roi_size;
                end_w = start_w + self._crop_size

                # if there is some remainder result for cropping, change start_h and start_w
                if end_h > in_pad_img.shape[0]:
                    start_h = in_pad_img.shape[0] - self._crop_size
                    end_h = in_pad_img.shape[0]
                if end_w > in_pad_img.shape[1]:
                    start_w = in_pad_img.shape[1] - self._crop_size
                    end_w = in_pad_img.shape[1]

                crop_img = in_pad_img[start_h: end_h, start_w: end_w]
                # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
                in_crop_imgs.append(crop_img)
                if end_w == in_pad_img.shape[1]:
                    break
            if end_h == in_pad_img.shape[0]:
                break
        return in_crop_imgs
    # def crop(self, in_img: np.ndarray) -> List:
    #     """
    #     Crop in_img to sub-img List, hint:self._roi_size is used in crop and stitch stage, please check the paper careful when read this code
    #     """
    #     self._in_img_shape = in_img.shape
    #     # Pad image before cropping
    #     in_pad_img = np.pad(in_img, self._overlap_size, mode='symmetric')  # 对称填充 各维度填充相同长度
    #     # calculate the number of cropping, which consider the influence of remainder
    #     h_pad_num = math.ceil(in_pad_img.shape[0] / (self._roi_size))  # 向上取整
    #     w_pad_num = math.ceil(in_pad_img.shape[1] / (self._roi_size))
    #     if in_pad_img.shape[0] % (self._roi_size) == 0:h_pad_num = h_pad_num - 1
    #     if in_pad_img.shape[1] % (self._roi_size) == 0:w_pad_num = w_pad_num - 1
    #     in_crop_imgs = []
    #     for i in range(h_pad_num):
    #         for j in range(w_pad_num):
    #             # overlap_crop, it is need to calculate the start of cropping
    #             start_h = i * self._roi_size;
    #             end_h = start_h + self._crop_size
    #             start_w = j * self._roi_size;
    #             end_w = start_w + self._crop_size
    #
    #             # if there is some remainder result for cropping, change start_h and start_w，不超出边缘
    #             if end_h > in_pad_img.shape[0]:
    #                 start_h = in_pad_img.shape[0] - self._crop_size
    #                 end_h = in_pad_img.shape[0]
    #             if end_w > in_pad_img.shape[1]:
    #                 start_w = in_pad_img.shape[1] - self._crop_size
    #                 end_w = in_pad_img.shape[1]
    #
    #             crop_img = in_pad_img[start_h: end_h, start_w: end_w]
    #             # print("cropping: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}, crop_shape={}".format(i, start_h, start_w, j, end_h, end_w, crop_img.shape))
    #             in_crop_imgs.append(crop_img)
    #     return in_crop_imgs

    def stitch(self, out_crop_imgs: List) -> np.ndarray:
        """
        Stitch sub-img List to whole out img
        """
        out_img = np.zeros(self._in_img_shape)

        # calculate the number of cropping, which consider the influence of remainder
        h_num = math.ceil(self._in_img_shape[0] / self._roi_size)
        w_num = math.ceil(self._in_img_shape[1] / self._roi_size)

        for img_idx, out_crop_img in enumerate(out_crop_imgs):
            roi_img = out_crop_img[self._overlap_size: self._overlap_size + self._roi_size,
                      self._overlap_size: self._overlap_size + self._roi_size]
            i = int(img_idx / w_num)
            j = int(img_idx - (i * w_num))
            start_h = int(i * self._roi_size);
            end_h = start_h + self._roi_size
            start_w = int(j * self._roi_size);
            end_w = start_w + self._roi_size

            if end_h > self._in_img_shape[0]:
                start_h = self._in_img_shape[0] - self._roi_size
                end_h = self._in_img_shape[0]
            if end_w > self._in_img_shape[1]:
                start_w = self._in_img_shape[1] - self._roi_size
                end_w = self._in_img_shape[1]
                # print("stitching: i={}, start_h={}, start_w={}, j={}, end_h={}, end_w={}".format(i, start_h, start_w, j, end_h, end_w))
            out_img[start_h: end_h, start_w: end_w] = roi_img
        return out_img


class TestTimeAug():
    """
    Test Time Augmentation
    """

    def __init__(self, aug_type: int = 0):
        self._aug_type = aug_type  # 弹性变换，0: no tta; 1: 4 variant(rotation), 2: eight variant(rotation and flip)

    def aug_input(self, in_imgs: List):
        """
        Aug sub-imgs List and return a new List
        """
        in_aug_imgs = []
        for in_img in in_imgs:
            if self._aug_type >= 0:
                in_aug_imgs.append(in_img.copy())
            if self._aug_type >= 1:  # rotation 4 variants for one input
                in_aug_imgs.append(
                    np.rot90(in_img, 1).copy())  # if there is no copy, it will introduce error in network inference
                in_aug_imgs.append(np.rot90(in_img, 2).copy())  # 逆时针旋转90*k度
                in_aug_imgs.append(np.rot90(in_img, 3).copy())
            if self._aug_type >= 2:  # rotation and flip 8 variants for one input
                in_aug_imgs.append(np.flipud(in_img).copy())  # upper and down flip
                in_aug_imgs.append(np.fliplr(in_img).copy())  # left and right flip
                in_aug_imgs.append(np.flipud(np.rot90(in_img, 3)).copy())  # upper and down flip for 270 rotation
                in_aug_imgs.append(np.fliplr(np.rot90(in_img, 3)).copy())  # left and right flip for 270 rotation
        return in_aug_imgs

    def merge_out(self, out_aug_imgs: List):
        """
        Merge aug List(Mean) and return sub-imgs List
        """
        out_imgs = []
        tta_num = self._aug_type * 4
        idx = 0
        while idx < len(out_aug_imgs):
            if self._aug_type >= 0:
                sum_imgs = out_aug_imgs[idx].copy()
            if self._aug_type >= 1:
                sum_imgs = sum_imgs + np.rot90(out_aug_imgs[idx + 1], -1)
                sum_imgs = sum_imgs + np.rot90(out_aug_imgs[idx + 2], -2)
                sum_imgs = sum_imgs + np.rot90(out_aug_imgs[idx + 3], -3)
            if self._aug_type >= 2:
                sum_imgs = sum_imgs + np.flipud(out_aug_imgs[idx + 4])
                sum_imgs = sum_imgs + np.fliplr(out_aug_imgs[idx + 5])
                sum_imgs = sum_imgs + np.rot90(np.flipud(out_aug_imgs[idx + 6]), -3)
                sum_imgs = sum_imgs + np.rot90(np.fliplr(out_aug_imgs[idx + 7]), -3)
            if self._aug_type == 0:
                out_imgs.append(sum_imgs)
                idx = idx + 1
            elif self._aug_type >= 1:
                out_imgs.append(sum_imgs / tta_num)
                idx = idx + tta_num
        return out_imgs

##UNet单图多图分割
class SegThread(QThread):
    cropBegin = pyqtSignal()
    segBegin = pyqtSignal(int)
    segSignal = pyqtSignal(int)
    transBegin = pyqtSignal()
    indexTrans = pyqtSignal(int)
    completeSignal = pyqtSignal(np.ndarray)
    runtimeSignal = pyqtSignal()
    grayError = pyqtSignal()
    finish = pyqtSignal()
    def __init__(self, input_size,  pth_address, overlap_size, batch_size, aug_type, mean, std,
                 use_post_process,
                 indexslot,cropbeginslot, segbeginslot, segslot, transbeginslot, completeslot,OTSUfinish,img_path=None,dir=None,in_image=np.array([]),
                 filedict = {},savedict = {}):
        super(SegThread, self).__init__()
        self.widget = QWidget()
        self.img_path = img_path
        self.pth_address = pth_address
        self.input_size = input_size
        self.overlap_size = overlap_size
        self.batch_size = batch_size
        self.aug_type = aug_type
        self.mean = mean
        self.std = std
        self.use_post_process = use_post_process
        self.in_img = in_image
        self.outdir = dir
        self.filedict = filedict
        self.savedict = savedict
        self.flag = 1
        self.completeSignal.connect(completeslot)
        self.indexTrans.connect(indexslot)
        # processbar singal
        self.transBegin.connect(transbeginslot)
        self.cropBegin.connect(cropbeginslot)
        self.segBegin.connect(segbeginslot)
        self.segSignal.connect(segslot)
        self.finish.connect(OTSUfinish)
        # logging.info(self.filename)

    def run(self):
        try:
            if self.flag != 1:
                return
            else:

                self.gpu_detector = GpuDetector()
                self.net_inference = NetInference(self.gpu_detector, input_size=self.input_size, overlap_size=self.overlap_size,
                                             batch_size=self.batch_size, aug_type=self.aug_type,
                                             mean=self.mean, std=self.std, pth_address=self.pth_address,
                                             use_post_process=self.use_post_process)
                print("1")
                if not self.filedict:
                    if self.in_img ==None or self.in_img.size == 0:
                        self.in_img = load_img(self.img_path)
                    out_img = self._forward_one_image(self.in_img)

                    if type(out_img) is np.ndarray and out_img.size != 0:
                        self.completeSignal.emit(out_img)

                    else:
                        self.completeSignal.emit(np.array([]))

                else:
                    self.inference()
        except AssertionError as e:
            self.grayError.emit()
            self.stop()
            print(e)
        except RuntimeError as e:
            self.runtimeSignal.emit()
            self.stop()
            print(e)
        except Exception as e:
            self.runtimeSignal.emit()
            self.stop()
            print(e)

    def stop(self):
        self.flag = 0

    def _forward_one_image(self, in_ori_img: np.ndarray) -> np.ndarray:
        """
        Forward one image, return segmentation result, some errers will be raised
        """

        # For overlap tile strategy and tta, the sub imgs are in List Type
        overlap_tile = OverlapTile(crop_size=self.input_size, overlap_size=self.overlap_size)  # 初始化平铺策略的参数
        tta = TestTimeAug(aug_type=self.aug_type)  # 初始化弹性变换方式
        self._model = self.net_inference._model
        self._model.eval()  # **模型必有
        with torch.no_grad():
            # overlap-tile strategy, crop img into sub-imgs in List and revert after network inference
            self.cropBegin.emit()
            in_imgs = overlap_tile.crop(in_ori_img)  # 得到裁剪块大小的图片序列
            in_imgs = tta.aug_input(in_imgs)  # 弹性变换后的图片
            out_imgs = []
            try:
                img_idx = 0
                h, w = in_imgs[0].shape

                self.segBegin.emit(len(in_imgs))
                while img_idx < len(in_imgs):
                    if self.flag == 1:
                        self.segSignal.emit(img_idx)
                        iter_batch_size = self.batch_size
                        if img_idx + iter_batch_size > len(in_imgs):
                            # if the len(in_imgs) can not be divided by batch_size, the batch size of last itereation should be
                            # len(in_imgs) - img_idx
                            iter_batch_size = len(in_imgs) - img_idx

                        in_tensor = torch.zeros((iter_batch_size, 1, h, w))
                        for b_idx in range(iter_batch_size):
                            in_tensor[b_idx, :, :, :] = self.net_inference._z_score_norm(in_imgs[img_idx + b_idx])

                        gpu_status, gpu_idx_in_use = self.gpu_detector._get_current_gpu()
                        if gpu_status:
                            in_tensor = in_tensor.cuda(gpu_idx_in_use)
                        out_tensor = self._model.forward(in_tensor)  # b,c,w,h c=2
                        out_tensor = F.softmax(out_tensor, dim=1)  # 将张量的每个元素缩放到（0,1）区间且和为1，dim=1按行
                        if gpu_status:
                            out_tensor = out_tensor.cpu()

                        for b_idx in range(iter_batch_size):
                            # the out data is presented by List again
                            out_imgs.append(out_tensor[b_idx, :, :, :].numpy().transpose((1, 2, 0)))  # 顺序改变从012到120
                        img_idx = img_idx + iter_batch_size

                        torch.cuda.empty_cache()

                    else:
                        print('stop')
                        return
            except RuntimeError as e:

                raise e                  # GPU or CPU memory is out of use
            else:
                out_imgs = tta.merge_out(out_imgs)  # merge different tta results.
                self.transBegin.emit()
                for out_idx in range(len(out_imgs)):
                    # transform type and post process

                    out_imgs[out_idx] = np.argmax(out_imgs[out_idx], axis=2)  # 返回沿轴axis最大值的索引
                    if self.use_post_process:
                        out_imgs[out_idx] = self.net_inference._post_process(out_imgs[out_idx])

                out_img = overlap_tile.stitch(out_imgs).clip(0, 1).astype(np.uint8)

        return out_img
    def inference(self):

        # For overlap tile strategy and tta, the sub imgs are in List Type
        lis = []
        if self.savedict:
            adict = self.savedict
        else:
            adict = self.filedict
        for ke in adict.keys():
            lis.append(ke)
        for i in range(len(lis)):

            self.indexTrans.emit(i)
            if self.flag == 1:
                text = lis[i]
                dirname = adict[text]
                ori_path = os.path.join(dirname, text)
                in_ori_img = load_img(ori_path)
                try:
                    assert in_ori_img.ndim == 2, "The input image should be gray scale"
                except AssertionError as err:
                    raise(err)
                overlap_tile = OverlapTile(crop_size=self.input_size, overlap_size=self.overlap_size)  # 初始化平铺策略的参数
                tta = TestTimeAug(aug_type=self.aug_type)  # 初始化弹性变换方式
                self._model = self.net_inference._model
                self._model.eval()  # **模型必有
                with torch.no_grad():
                    # overlap-tile strategy, crop img into sub-imgs in List and revert after network inference

                    in_imgs = overlap_tile.crop(in_ori_img)  # 得到裁剪块大小的图片序列
                    in_imgs = tta.aug_input(in_imgs)  # 弹性变换后的图片
                    out_imgs = []
                    try:
                        img_idx = 0
                        h, w = in_imgs[0].shape

                        while img_idx < len(in_imgs):
                            if self.flag == 1:

                                iter_batch_size = self.batch_size
                                if img_idx + iter_batch_size > len(in_imgs):
                                    # if the len(in_imgs) can not be divided by batch_size, the batch size of last itereation should be
                                    # len(in_imgs) - img_idx
                                    iter_batch_size = len(in_imgs) - img_idx

                                in_tensor = torch.zeros((iter_batch_size, 1, h, w))
                                for b_idx in range(iter_batch_size):
                                    in_tensor[b_idx, :, :, :] = self.net_inference._z_score_norm(in_imgs[img_idx + b_idx])

                                gpu_status, gpu_idx_in_use = self.gpu_detector._get_current_gpu()
                                if gpu_status:
                                    in_tensor = in_tensor.cuda(gpu_idx_in_use)
                                out_tensor = self._model.forward(in_tensor)  # b,c,w,h c=2
                                out_tensor = F.softmax(out_tensor, dim=1)  # 将张量的每个元素缩放到（0,1）区间且和为1，dim=1按行
                                if gpu_status:
                                    out_tensor = out_tensor.cpu()

                                for b_idx in range(iter_batch_size):
                                    # the out data is presented by List again
                                    out_imgs.append(out_tensor[b_idx, :, :, :].numpy().transpose((1, 2, 0)))  # 顺序改变从012到120
                                img_idx = img_idx + iter_batch_size

                                torch.cuda.empty_cache()
                            else:
                                print('stop')
                                return
                    except RuntimeError as e:
                        raise e               # GPU or CPU memory is out of use
                    else:
                        out_imgs = tta.merge_out(out_imgs)  # merge different tta results.

                        for out_idx in range(len(out_imgs)):
                            # transform type and post process

                            out_imgs[out_idx] = np.argmax(out_imgs[out_idx], axis=2)  # 返回沿轴axis最大值的索引
                            if self.use_post_process:
                                out_imgs[out_idx] = self.net_inference._post_process(out_imgs[out_idx])

                        out_img = overlap_tile.stitch(out_imgs).clip(0, 1).astype(np.uint8)
                        file_name = os.path.splitext(ori_path)[0] + '_label.png'
                        name = file_name.split("\\")[-1]

                        outdir = self.outdir.replace('/', '\\')
                        file_name = os.path.join(outdir, name)
                        un = out_img * 255
                        d_img = un.astype(np.uint8)

                        shrink = cv2.cvtColor(d_img, cv2.COLOR_BGR2RGB)
                        seg = QImage(shrink.data,
                                          shrink.shape[1],
                                          shrink.shape[0],
                                          QImage.Format_RGB888)
                        image2 = seg.convertToFormat(QImage.Format_ARGB32)
                        image2.save(file_name, "PNG")
            else:
                print('stop')
                return
        self.finish.emit()