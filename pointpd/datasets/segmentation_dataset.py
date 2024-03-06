"""
Segmentation Datasets

Author: Shoutong Luo
Please cite our work if the code is helpful to you.
"""

import os
import glob
import numpy as np
import torch
from copy import deepcopy
from torch.utils.data import Dataset
from collections.abc import Sequence
import json

import errno
import pickle
from pointpd.utils.logger import get_root_logger
from .builder import DATASETS, build_dataset
from .transform import Compose, TRANSFORMS
from .helper import *
import copy

'''
def write_dataset_split(split_path):
    with open(split_path, 'w') as f:
        json.dump(data, f)
'''

def load_dataset_split(split_path):
    if not os.path.exists(split_path):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), split_path)
    with open(split_path, 'r') as f:
        split_info = json.load(f)
    return split_info


class SegmentationPartitionDataset(Dataset):
    def __init__(
        self,
        split="train",
        split_file="default",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        feat_keys = ["color"],
        segment_key = "semantic_gt",
        preload_data_to_memory = False,
        preload_partition_info = False,
        partition_size = 0.04,
        colormap = None,
        loop=1,
        **kwargs,
    ):
        super(SegmentationPartitionDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.split_file = split_file
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.test_cfg = test_cfg if test_mode else None
        self.feat_keys = feat_keys
        self.segment_key = segment_key
        self.preload_data_to_memory = preload_data_to_memory
        self.preload_partition_info = preload_partition_info
        self.partition_size = partition_size
        self.colormap = colormap
        self.class_names = []
        if self.colormap is not None:
            colorarray = []
            for k, v in self.colormap.items():
                self.class_names.append(k)
                colorarray.append(v)
        self.colormap = np.array(colorarray) / 255

        if test_mode:
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop is not None
                else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        self.preload_data = self.load_data() if self.preload_data_to_memory else None
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )
    
    def load_data(self):
        '''
        Pre-load some scenes with a high number of points into memory to save training time
        Turn off this feature if there is not enough memory, just set preload_data_to_memory to False
        '''
        data = []
        for dl in self.data_list:
            data.append(torch.load(dl))
        return data

    def get_data_list(self):
        split_path = os.path.join(self.data_root, self.split_file + ".json")
        split_info = load_dataset_split(split_path)

        filenames = split_info[self.split]

        data_list = [
            os.path.join(self.data_root, "data", x + ".pth") for x in filenames
        ]
        '''
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        '''
        return data_list

    def get_data(self, idx):
        if self.preload_data_to_memory:
            data = self.preload_data[idx % len(self.data_list)]
        else:
            data = torch.load(self.data_list[idx % len(self.data_list)])

        coord = data["coord"]

        if self.segment_key in data.keys():
            segment = data[self.segment_key].reshape([-1])
        else:
            segment = np.ones(coord.shape[0]) * -1

        data_dict = dict(coord=coord, segment=segment.astype(np.int64))

        for fk in self.feat_keys:
            data_dict[fk] = data[fk].reshape((-1, 1)) if len(data[fk].shape) == 1 else data[fk] 
        
        if self.preload_partition_info:
            # Load the partition information in advance, because the partition of point clouds with tens of millions of points takes a very long time (i.e. GridSample), 
            filename = self.get_data_name(idx)
            partition_file = os.path.join(self.data_root, "{:.02f}".format(self.partition_size), f"{filename}_partition.pkl")
            assert os.path.exists(partition_file), "{} not found!".format(partition_file)

            with open(partition_file, 'rb') as f:
                partition_dict = pickle.load(f)
                idx_sort, count = partition_dict["idx_sort"], partition_dict["count"]
        else:
            idx_sort, count = None, None
        
        data_dict["idx_sort"] = idx_sort
        data_dict["count"] = count

        #print(data_dict.keys(), self.feat_keys)

        return data_dict
    
    def get_full_cloud(self, idx):
        return self.get_data(idx)['coord']

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)

        keys = ['coord', 'segment'] + [k for k in self.feat_keys]
        data_dict = Partition(grid_size=self.partition_size, hash_type="fnv", mode="train", keys=keys)(data_dict)
        
        del data_dict["idx_sort"]
        del data_dict["count"]

        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        '''
        Memory efficient implementation. 
        Let the network process the data once per load instead of loading it all. 
        For Semantic3D and SensatUrban, a single scene may require more than 100G of memory to load all the way through
        '''
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)

        for ia, aug in enumerate(self.aug_transform):
            data = aug(deepcopy(data_dict))
            keys = ['coord', 'segment'] + [k for k in self.feat_keys]
            data_part_list = Partition(grid_size=self.partition_size, hash_type="fnv", mode="test", keys=keys)(data)
            for idp, data_part in enumerate(data_part_list):
                del data_part["idx_sort"]
                del data_part["count"]
                if self.test_crop:
                    data_crop = self.test_crop(data_part)
                else:
                    data_crop = [data_part]
                data_crop = data_crop
                for idc, dc in enumerate(data_crop):  
                    dc = self.post_transform(copy.deepcopy(dc))

                    self.aug_num = len(self.aug_transform)
                    self.cur_aug_idx = ia
                    self.data_part_num = len(data_part_list)
                    self.cur_dp_idx = idp
                    self.data_crop_num = len(data_crop)
                    self.cur_dc_idx = idc
                    input_dict = dict(
                        fragment=dc, segment=segment, complete=((ia == self.aug_num - 1) and (idp == self.data_part_num - 1) and (idc == self.data_crop_num - 1))
                        #fragment=dc, segment=segment, complete=True
                    )
                    yield input_dict
        return data_dict

    def __getitem__(self, idx):
        if not self.test_mode:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


class SegmentationSubCloudDataset(Dataset):
    def __init__(
        self,
        split="train",
        split_file="default",
        data_root="data/dataset",
        transform=None,
        test_mode=False,
        test_cfg=None,
        feat_keys = ["color"],
        segment_key = "semantic_gt",
        preload_subcloud_info = False,
        voxel_size = 0.04,
        colormap = None,
        loop=1,
        **kwargs
    ):
        super(SegmentationSubCloudDataset, self).__init__()
        self.data_root = data_root
        self.split = split
        self.split_file = split_file
        self.transform = Compose(transform)
        self.loop = (
            loop if not test_mode else 1
        )  # force make loop = 1 while in test mode
        self.test_mode = test_mode
        self.feat_keys = feat_keys
        self.segment_key = segment_key
        self.voxel_size = voxel_size
        self.test_cfg = test_cfg if test_mode else None
        self.preload_subcloud_info = preload_subcloud_info
        self.colormap = colormap
        self.class_names = []
        if self.colormap is not None:
            colorarray = []
            for k, v in self.colormap.items():
                self.class_names.append(k)
                colorarray.append(v)
        self.colormap = np.array(colorarray) / 255

        if test_mode:
            self.test_crop = (
                TRANSFORMS.build(self.test_cfg.crop)
                if self.test_cfg.crop is not None
                else None
            )
            self.post_transform = Compose(self.test_cfg.post_transform)
            self.aug_transform = [Compose(aug) for aug in self.test_cfg.aug_transform]

        self.data_list = self.get_data_list()
        self.sub_tree = []
        logger = get_root_logger()
        logger.info(
            "Totally {} x {} samples in {} set.".format(
                len(self.data_list), self.loop, split
            )
        )

    def get_data_list(self):
        split_path = os.path.join(self.data_root, self.split_file + ".json")
        split_info = load_dataset_split(split_path)

        filenames = split_info[self.split]

        data_list = [
            os.path.join(self.data_root, "data", x + ".pth") for x in filenames
        ]
        '''
        if isinstance(self.split, str):
            data_list = glob.glob(os.path.join(self.data_root, self.split, "*.pth"))
        elif isinstance(self.split, Sequence):
            data_list = []
            for split in self.split:
                data_list += glob.glob(os.path.join(self.data_root, split, "*.pth"))
        else:
            raise NotImplementedError
        '''
        return data_list
        

    def get_data(self, idx):
        if self.preload_subcloud_info:
            filename = self.get_data_name(idx)
            subcloud_file = os.path.join(self.data_root, "{:.02f}".format(self.voxel_size), f"{filename}_subcloud.pkl")
            assert os.path.exists(subcloud_file), "{} not found!".format(subcloud_file)

            with open(subcloud_file, 'rb') as f:
                subcloud_dict = pickle.load(f)
                data_dict = dict(coord=subcloud_dict['sub_coord'], segment=subcloud_dict['sub_{}'.format(self.segment_key)].reshape(-1).astype(np.int64))
                for key in self.feat_keys:
                    data_dict[key] = subcloud_dict["sub_{}".format(key)]
                
                #if "sub_tree" in subcloud_dict.keys():
                    #self.sub_tree.append(subcloud_dict["sub_tree"])
        else:
            data = torch.load(self.data_list[idx % len(self.data_list)])
            coord = data["coord"]
        
            if self.segment_key in data.keys():
                segment = data[self.segment_key].reshape([-1])
            else:
                segment = np.ones(coord.shape[0]) * -1
            
            feats = np.column_stack([data[key] for key in self.feat_keys]).astype(np.float32)

            sub_coord, sub_feats, sub_segment = grid_sub_sampling(coord, feats, segment, grid_size=self.voxel_size)
            data_dict = dict(coord=sub_coord, segment=sub_segment, full_cloud=coord, full_segment=segment)

            column_start = 0
            for key in self.feat_keys:
                column_size = 1 if len(data[key].shape) == 1 else data[key].shape[1]
                data_dict[key] = sub_feats[:, column_start: column_start + column_size]
                column_start += column_size

        return data_dict

    def get_data_name(self, idx):
        return os.path.basename(self.data_list[idx % len(self.data_list)]).split(".")[0]

    def get_full_cloud(self, idx):
        data_dict = self.get_data(idx)
        if "full_cloud" not in data_dict.keys():
            data = torch.load(self.data_list[idx % len(self.data_list)])
            full_coord = data["coord"]
            full_segment = data[self.segment_key]
        else:
            full_coord = data_dict["full_coord"]
            full_segment = data_dict["full_segment"]
        return data_dict["coord"], full_coord, full_segment

    def prepare_train_data(self, idx):
        # load data
        data_dict = self.get_data(idx)
        data_dict = self.transform(data_dict)
        return data_dict

    def prepare_test_data(self, idx):
        data_dict = self.get_data(idx)
        segment = data_dict.pop("segment")
        data_dict = self.transform(data_dict)

        for ia, aug in enumerate(self.aug_transform):
            data_part = aug(deepcopy(data_dict))
            if self.test_crop:
                data_crop = self.test_crop(data_part)
            else:
                data_part["index"] = np.arange(data_part["coord"].shape[0])
                data_crop = [data_part]
            data_crop = data_crop
            for idc, dc in enumerate(data_crop):  
                dc = self.post_transform(copy.deepcopy(dc))

                self.aug_num = len(self.aug_transform)
                self.cur_aug_idx = ia
                self.data_crop_num = len(data_crop)
                self.cur_dc_idx = idc
                input_dict = dict(
                    fragment=dc, segment=segment, complete=((ia == self.aug_num - 1) and (idc == self.data_crop_num - 1))
                    #fragment=dc, segment=segment, complete=True
                )
                yield input_dict

    def __getitem__(self, idx):
        if not self.test_mode:
            return self.prepare_train_data(idx)

    def __len__(self):
        return len(self.data_list) * self.loop


@DATASETS.register_module()
class S3DISPartitionDataset(SegmentationPartitionDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color"],
            segment_key = "semantic_gt",
            preload_partition_info = False,
            partition_size = 0.04,
            colormap = S3DIS_COLORMAP,
        )
        kkargs.update(kwargs)
        super(S3DISPartitionDataset, self).__init__(**kkargs)

@DATASETS.register_module()
class S3DISSubCloudDataset(SegmentationSubCloudDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color"],
            segment_key = "semantic_gt",
            preload_subcloud_info = False,
            voxel_size = 0.04,
            colormap = S3DIS_COLORMAP,
        )
        kkargs.update(kwargs)
        super(S3DISSubCloudDataset, self).__init__(**kkargs)

@DATASETS.register_module()
class ScannetPartitionDataset(SegmentationPartitionDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color", "normal"],
            segment_key = "semantic_gt20",
            preload_partition_info = False,
            partition_size = 0.02,
            colormap = SCANNET_COLORMAP,
        )
        kkargs.update(kwargs)
        super(ScannetPartitionDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class ScannetSubCloudDataset(SegmentationSubCloudDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color", "normal"],
            segment_key = "semantic_gt20",
            preload_subcloud_info = False,
            voxel_size = 0.02,
            colormap = SCANNET_COLORMAP,
        )
        kkargs.update(kwargs)
        super(ScannetSubCloudDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class Scannet200PartitionDataset(SegmentationPartitionDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color", "normal"],
            segment_key = "semantic_gt200",
            preload_partition_info = False,
            partition_size = 0.02,
            colormap = SCANNET200_COLORMAP,
        )
        kkargs.update(kwargs)
        super(ScannetPartitionDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class Scannet200SubCloudDataset(SegmentationSubCloudDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color", "normal"],
            segment_key = "semantic_gt200",
            preload_subcloud_info = False,
            voxel_size = 0.02,
            colormap = SCANNET200_COLORMAP,
        )
        kkargs.update(kwargs)
        super(ScannetSubCloudDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class SemanticKittiPartitionDataset(SegmentationPartitionDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["strength"],
            segment_key = "semantic_gt",
            preload_partition_info = False,
            partition_size = 0.2,
            colormap = SEMANTICKITTI_COLORMAP,
        )
        kkargs.update(kwargs)
        super(SemanticKittiPartitionDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class SemanticKittiSubCloudDataset(SegmentationSubCloudDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["strength"],
            segment_key = "semantic_gt",
            preload_subcloud_info = False,
            voxel_size = 0.2,
            colormap = SEMANTICKITTI_COLORMAP,
        )
        kkargs.update(kwargs)
        super(SemanticKittiSubCloudDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class Semantic3DPartitionDataset(SegmentationPartitionDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color"],
            segment_key = "semantic_gt",
            preload_data_to_memory = True,
            preload_partition_info = True,
            partition_size = 0.2,
            colormap = SEMANTIC3D_COLORMAP,
        )
        kkargs.update(kwargs)
        super(Semantic3DPartitionDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class Semantic3DSubCloudDataset(SegmentationSubCloudDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color"],
            segment_key = "semantic_gt",
            preload_subcloud_info = True,
            voxel_size = 0.2,
            colormap = SEMANTIC3D_COLORMAP,
        )
        kkargs.update(kwargs)
        super(Semantic3DSubCloudDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class SensatUrbanPartitionDataset(SegmentationPartitionDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color"],
            segment_key = "semantic_gt",
            preload_data_to_memory = True,
            preload_partition_info = True,
            partition_size = 0.4,
            colormap = SENSATURBAN_COLORMAP,
        )
        kkargs.update(kwargs)
        super(SensatUrbanPartitionDataset, self).__init__(
            **kkargs
        )

@DATASETS.register_module()
class SensatUrbanSubCloudDataset(SegmentationSubCloudDataset):
    def __init__(self, **kwargs):
        kkargs = dict(
            feat_keys = ["color"],
            segment_key = "semantic_gt",
            preload_subcloud_info = True,
            voxel_size = 0.4,
            colormap = SENSATURBAN_COLORMAP,
        )
        kkargs.update(kwargs)
        super(SensatUrbanSubCloudDataset, self).__init__(
            **kkargs
        )