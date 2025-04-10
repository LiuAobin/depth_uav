import numpy as np
import skimage.transform
import PIL.Image as pil
from system.datasets.mono_dataset import MonoDataset
from system.utils import generate_depth_map,read_calib_file

class KittiDataset(MonoDataset):
    def __init__(self, cfg,stage='train'):
        super(KittiDataset,self).__init__(cfg,stage)
        # self.K = np.array([[0.58, 0, 0.5, 0],
        #                    [0, 1.92, 0.5, 0],
        #                    [0, 0, 1, 0],
        #                    [0, 0, 0, 1]], dtype=np.float32)
        self.K = np.array([[0.58, 0, 0.5],
                           [0, 1.92, 0.5],
                           [0, 0, 1]], dtype=np.float32)
        self.full_res_shape = (1242, 375)
        self.side_map = {"2":2,"3":3,"l":2,"r":3}

    def get_depth(self,folder,frame_index,side,do_flip):
        calib_path = self.data_path.joinpath(folder.split('/')[0])
        velo_filename = self.data_path.joinpath(folder,
                                                "velodyne_points","data",
                                                "{:010d}.bin".format(int(frame_index)))
        depth_gt = generate_depth_map(calib_path, velo_filename,self.side_map[side])
        depth_gt = skimage.transform.resize(
            depth_gt,self.full_res_shape[::-1],
            order=0, preserve_range=True, mode='constant'
        )
        if do_flip:
            depth_gt = np.fliplr(depth_gt)

        return depth_gt

    def get_color(self,folder,frame_index,side,do_flip):
        image_path = self.get_image_path(folder,frame_index,side)
        color = self.loader(image_path)
        if do_flip:
            color = color.transpose(pil.FLIP_LEFT_RIGHT)
        return color

    def get_intrinsics(self,folder,frame_index,side,do_flip,height,width):
        K = self.K.copy()
        K[0, :] *= self.width
        K[1, :] *= self.height
        K3x3 = K[:3, :3]
        return K,K3x3
        # calib_path = self.data_path.joinpath(folder.split('/')[0])
        # cam2cam = read_calib_file(calib_path.joinpath('calib_cam_to_cam.txt'))
        # # 读取图像原始分辨率（注意 KITTI 原始图像是 1242x375，或从标定文件中读取）
        # image_width = int(cam2cam["S_rect_0" + str(self.side_map[side])][0])
        # image_height = int(cam2cam["S_rect_0" + str(self.side_map[side])][1])
        # P_rect = cam2cam['P_rect_0' + str(self.side_map[side])].reshape(3, 4)
        # # 分离内参矩阵 K（左上 3x3 部分）
        # K = P_rect[:3, :3]
        # K[0, :] *= (width / image_width)
        # K[1, :] *= (height / image_height)
        # K_full = np.eye(4, dtype=np.float32)
        # K_full[:3, :3] = K
        # # 可选：do_flip 时需要水平翻转 cx
        # if do_flip:
        #     K_full[0, 2] = 1.0 - K_full[0, 2]
        # return K_full,K

    def get_image_path(self,folder,frame_index,side):
        f_str="{:010d}{}".format(frame_index,self.img_ext)
        image_path = self.data_path.joinpath(folder,'image_0{}/data'.format(self.side_map[side]),f_str)
        return image_path
