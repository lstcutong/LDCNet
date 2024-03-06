try:
    import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
except:
    pass
import torch
import glob
import os
from os.path import join, exists, dirname, abspath
import numpy as np
import pickle
from sklearn.neighbors import KDTree
from pointpd.datasets.preprocessing.scannet.meta_data.scannet200_constants import (
    CLASS_LABELS_20, CLASS_LABELS_200,
)
import sys

def grid_sub_sampling(points, features=None, labels=None, grid_size=0.1, verbose=0):
    """
    CPP wrapper for a grid sub_sampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param grid_size: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: sub_sampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=grid_size, verbose=verbose)
    elif labels is None:
        return cpp_subsampling.compute(points, features=features, sampleDl=grid_size, verbose=verbose)
    elif features is None:
        return cpp_subsampling.compute(points, classes=labels, sampleDl=grid_size, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=grid_size,
                                           verbose=verbose)

class Partition(object):
    def __init__(
        self,
        grid_size=0.05,
        hash_type="fnv",
        mode="train",
        keys=("coord", "color", "normal", "segment"),
        return_discrete_coord=False,
        return_min_coord=False,
        return_displacement=False,
        project_displacement=False,
    ):
        self.grid_size = grid_size
        self.hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "test"]
        self.mode = mode
        self.keys = keys
        self.return_discrete_coord = return_discrete_coord
        self.return_min_coord = return_min_coord
        self.return_displacement = return_displacement
        self.project_displacement = project_displacement

    def __call__(self, data_dict):
        assert "coord" in data_dict.keys()
        scaled_coord = data_dict["coord"] / np.array(self.grid_size)
        discrete_coord = np.floor(scaled_coord).astype(int)
        min_coord = discrete_coord.min(0) * np.array(self.grid_size)
        discrete_coord -= discrete_coord.min(0)
        if data_dict["idx_sort"] is None:
            key = self.hash(discrete_coord)
            idx_sort = np.argsort(key)
            key_sort = key[idx_sort]
            _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)
        else:
            idx_sort, count = data_dict["idx_sort"], data_dict["count"]
        if self.mode == "train":  # train mode
            idx_select = (
                np.cumsum(np.insert(count, 0, 0)[0:-1])
                + np.random.randint(0, count.max(), count.size) % count
            )
            idx_unique = idx_sort[idx_select]
            if "sampled_index" in data_dict:
                # for ScanNet data efficient, we need to make sure labeled point is sampled.
                idx_unique = np.unique(
                    np.append(idx_unique, data_dict["sampled_index"])
                )
                mask = np.zeros_like(data_dict["segment"]).astype(np.bool)
                mask[data_dict["sampled_index"]] = True
                data_dict["sampled_index"] = np.where(mask[idx_unique])[0]
            if self.return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            if self.return_displacement:
                displacement = (
                    scaled_coord - discrete_coord - 0.5
                )  # [0, 1] -> [-0.5, 0.5] displacement to center
                if self.project_displacement:
                    displacement = np.sum(
                        displacement * data_dict["normal"], axis=-1, keepdims=True
                    )
                data_dict["displacement"] = displacement[idx_unique]
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict

        elif self.mode == "test":  # test mode
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                if self.return_displacement:
                    displacement = (
                        scaled_coord - discrete_coord - 0.5
                    )  # [0, 1] -> [-0.5, 0.5] displacement to center
                    if self.project_displacement:
                        displacement = np.sum(
                            displacement * data_dict["normal"], axis=-1, keepdims=True
                        )
                    data_dict["displacement"] = displacement[idx_part]
                for key in data_dict.keys():
                    if key in self.keys:
                        data_part[key] = data_dict[key][idx_part]
                    else:
                        data_part[key] = data_dict[key]
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    @staticmethod
    def ravel_hash_vec(arr):
        """
        Ravel the coordinates after subtracting the min coordinates.
        """
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    @staticmethod
    def fnv_hash_vec(arr):
        """
        FNV64-1A
        """
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(
            arr.shape[0], dtype=np.uint64
        )
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr


S3DIS_COLORMAP = {
    "ceiling": [231, 231, 91],
    "floor": [103, 157, 200],
    "wall": [177, 116, 76],
    "beam": [236, 150, 130],
    "column": [89, 164, 149],
    "window": [80, 175, 70],
    "door": [108, 136, 66],
    "table": [78, 78, 75],
    "chair": [41, 44, 104],
    "sofa": [87, 40, 96],
    "bookcase": [217, 49, 50],
    "board": [85, 109, 115],
    "clutter": [234, 234, 230]
}

ROCKERY_COLORMAP = {
    "building": [1,158,213],
    "plant": [255,94, 72],
    "rock": [179, 197, 135],
    "pond": [217, 116, 43]
}

SEMANTIC3D_COLORMAP = {
    "manmade terrian": [192,192,192],
    "natural terrian": [0, 65, 14],
    "high vegetation": [0, 252, 52],
    "low vegetation": [252, 253, 57],
    "buildings": [234, 2, 25],
    "remaining hard scape": [131, 15, 169],
    "scanning artifacts": [0, 250, 246],
    "cars and trucks": [241, 5, 110]
}

SEMANTICKITTI_COLORMAP = {
    "car": [245, 150, 100],
    "bicycle": [245, 230, 100],
    "motorcycle": [150, 60, 30],
    "truck": [180, 30, 80],
    "other-vehicle": [255, 0, 0],
    "person": [30, 30, 255],
    "bicyclist": [200, 40, 255],
    "motorcyclist": [90, 30, 150],
    "road": [255, 0, 255],
    "parking": [255, 150, 255],
    "sidewalk": [75, 0, 75],
    "other-ground": [75, 0, 175],
    "building": [0, 200, 255],
    "fence": [50, 120, 255],
    "vegetation": [0, 175, 0],
    "trunk": [0, 60, 135],
    "terrain": [80, 240, 150],
    "pole": [150, 240, 255],
    "traffic-sign": [0, 0, 255],
}

SENSATURBAN_COLORMAP = {
    "ground":(130,146,102),
    "vegetation": (67,255,67),
    "building":(255,189,.67),
    "wall":(97,103,141),
    "bridge":(67,67,67),
    "parking":(67,67,255),
    "rail":(255,67,255),
    "car":(255,67,67),
    "footpath":(255,255,67),
    "bike":(67,255,255),
    "water":(67,208,255),
    "traffic road":(215,215,215),
    "street forniture":(133,102,137)
}

SCANNET_COLORMAP = {
    "wall":(174.0, 199.0, 232.0),
    "floor":(152.0, 223.0, 138.0),
    "cabinet":(31.0, 119.0, 180.0),
    "bed":(255.0, 187.0, 120.0),
    "chair":(188.0, 189.0, 34.0),
    "sofa":(140.0, 86.0, 75.0),
    "table":(255.0, 152.0, 150.0),
    "door":(214.0, 39.0, 40.0),
    "window":(197.0, 176.0, 213.0),
    "bookshelf":(148.0, 103.0, 189.0),
    "picture":(196.0, 156.0, 148.0),
    "counter":(23.0, 190.0, 207.0),
    "desk":(247.0, 182.0, 210.0),
    "curtain":(219.0, 219.0, 141.0),
    "refrigerator":(255.0, 127.0, 14.0),
    "shower curtain":(158.0, 218.0, 229.0),
    "toilet":(44.0, 160.0, 44.0),
    "sink":(112.0, 128.0, 144.0),
    "bathtub":(227.0, 119.0, 194.0),
    "otherfurniture":(82.0, 84.0, 163.0),
}

SCANNET200_COLORMAP = {
    'wall': (174.0, 199.0, 232.0),
    'chair': (188.0, 189.0, 34.0),
    'floor': (152.0, 223.0, 138.0),
    'table': (255.0, 152.0, 150.0),
    'door': (214.0, 39.0, 40.0),
    'couch': (91.0, 135.0, 229.0),
    'cabinet': (31.0, 119.0, 180.0),
    'shelf': (229.0, 91.0, 104.0),
    'desk': (247.0, 182.0, 210.0),
    'office chair': (91.0, 229.0, 110.0),
    'bed': (255.0, 187.0, 120.0),
    'pillow': (141.0, 91.0, 229.0),
    'sink': (112.0, 128.0, 144.0),
    'picture': (196.0, 156.0, 148.0),
    'window': (197.0, 176.0, 213.0),
    'toilet': (44.0, 160.0, 44.0),
    'bookshelf': (148.0, 103.0, 189.0),
    'monitor': (229.0, 91.0, 223.0),
    'curtain': (219.0, 219.0, 141.0),
    'book': (192.0, 229.0, 91.0),
    'armchair': (88.0, 218.0, 137.0),
    'coffee table': (58.0, 98.0, 137.0),
    'box': (177.0, 82.0, 239.0),
    'refrigerator': (255.0, 127.0, 14.0),
    'lamp': (237.0, 204.0, 37.0),
    'kitchen cabinet': (41.0, 206.0, 32.0),
    'towel': (62.0, 143.0, 148.0),
    'clothes': (34.0, 14.0, 130.0),
    'tv': (143.0, 45.0, 115.0),
    'nightstand': (137.0, 63.0, 14.0),
    'counter': (23.0, 190.0, 207.0),
    'dresser': (16.0, 212.0, 139.0),
    'stool': (90.0, 119.0, 201.0),
    'cushion': (125.0, 30.0, 141.0),
    'plant': (150.0, 53.0, 56.0),
    'ceiling': (186.0, 197.0, 62.0),
    'bathtub': (227.0, 119.0, 194.0),
    'end table': (38.0, 100.0, 128.0),
    'dining table': (120.0, 31.0, 243.0),
    'keyboard': (154.0, 59.0, 103.0),
    'bag': (169.0, 137.0, 78.0),
    'backpack': (143.0, 245.0, 111.0),
    'toilet paper': (37.0, 230.0, 205.0),
    'printer': (14.0, 16.0, 155.0),
    'tv stand': (196.0, 51.0, 182.0),
    'whiteboard': (237.0, 80.0, 38.0),
    'blanket': (138.0, 175.0, 62.0),
    'shower curtain': (158.0, 218.0, 229.0),
    'trash can': (38.0, 96.0, 167.0),
    'closet': (190.0, 77.0, 246.0),
    'stairs': (208.0, 49.0, 84.0),
    'microwave': (208.0, 193.0, 72.0),
    'stove': (55.0, 220.0, 57.0),
    'shoe': (10.0, 125.0, 140.0),
    'computer tower': (76.0, 38.0, 202.0),
    'bottle': (191.0, 28.0, 135.0),
    'bin': (211.0, 120.0, 42.0),
    'ottoman': (118.0, 174.0, 76.0),
    'bench': (17.0, 242.0, 171.0),
    'board': (20.0, 65.0, 247.0),
    'washing machine': (208.0, 61.0, 222.0),
    'mirror': (162.0, 62.0, 60.0),
    'copier': (210.0, 235.0, 62.0),
    'basket': (45.0, 152.0, 72.0),
    'sofa chair': (35.0, 107.0, 149.0),
    'file cabinet': (160.0, 89.0, 237.0),
    'fan': (227.0, 56.0, 125.0),
    'laptop': (169.0, 143.0, 81.0),
    'shower': (42.0, 143.0, 20.0),
    'paper': (25.0, 160.0, 151.0),
    'person': (82.0, 75.0, 227.0),
    'paper towel dispenser': (253.0, 59.0, 222.0),
    'oven': (240.0, 130.0, 89.0),
    'blinds': (123.0, 172.0, 47.0),
    'rack': (71.0, 194.0, 133.0),
    'plate': (24.0, 94.0, 205.0),
    'blackboard': (134.0, 16.0, 179.0),
    'piano': (159.0, 32.0, 52.0),
    'suitcase': (213.0, 208.0, 88.0),
    'rail': (64.0, 158.0, 70.0),
    'radiator': (18.0, 163.0, 194.0),
    'recycling bin': (65.0, 29.0, 153.0),
    'container': (177.0, 10.0, 109.0),
    'wardrobe': (152.0, 83.0, 7.0),
    'soap dispenser': (83.0, 175.0, 30.0),
    'telephone': (18.0, 199.0, 153.0),
    'bucket': (61.0, 81.0, 208.0),
    'clock': (213.0, 85.0, 216.0),
    'stand': (170.0, 53.0, 42.0),
    'light': (161.0, 192.0, 38.0),
    'laundry basket': (23.0, 241.0, 91.0),
    'pipe': (12.0, 103.0, 170.0),
    'clothes dryer': (151.0, 41.0, 245.0),
    'guitar': (133.0, 51.0, 80.0),
    'toilet paper holder': (184.0, 162.0, 91.0),
    'seat': (50.0, 138.0, 38.0),
    'speaker': (31.0, 237.0, 236.0),
    'column': (39.0, 19.0, 208.0),
    'bicycle': (223.0, 27.0, 180.0),
    'ladder': (254.0, 141.0, 85.0),
    'bathroom stall': (97.0, 144.0, 39.0),
    'shower wall': (106.0, 231.0, 176.0),
    'cup': (12.0, 61.0, 162.0),
    'jacket': (124.0, 66.0, 140.0),
    'storage bin': (137.0, 66.0, 73.0),
    'coffee maker': (250.0, 253.0, 26.0),
    'dishwasher': (55.0, 191.0, 73.0),
    'paper towel roll': (60.0, 126.0, 146.0),
    'machine': (153.0, 108.0, 234.0),
    'mat': (184.0, 58.0, 125.0),
    'windowsill': (135.0, 84.0, 14.0),
    'bar': (139.0, 248.0, 91.0),
    'toaster': (53.0, 200.0, 172.0),
    'bulletin board': (63.0, 69.0, 134.0),
    'ironing board': (190.0, 75.0, 186.0),
    'fireplace': (127.0, 63.0, 52.0),
    'soap dish': (141.0, 182.0, 25.0),
    'kitchen counter': (56.0, 144.0, 89.0),
    'doorframe': (64.0, 160.0, 250.0),
    'toilet paper dispenser': (182.0, 86.0, 245.0),
    'mini fridge': (139.0, 18.0, 53.0),
    'fire extinguisher': (134.0, 120.0, 54.0),
    'ball': (49.0, 165.0, 42.0),
    'hat': (51.0, 128.0, 133.0),
    'shower curtain rod': (44.0, 21.0, 163.0),
    'water cooler': (232.0, 93.0, 193.0),
    'paper cutter': (176.0, 102.0, 54.0),
    'tray': (116.0, 217.0, 17.0),
    'shower door': (54.0, 209.0, 150.0),
    'pillar': (60.0, 99.0, 204.0),
    'ledge': (129.0, 43.0, 144.0),
    'toaster oven': (252.0, 100.0, 106.0),
    'mouse': (187.0, 196.0, 73.0),
    'toilet seat cover dispenser': (13.0, 158.0, 40.0),
    'furniture': (52.0, 122.0, 152.0),
    'cart': (128.0, 76.0, 202.0),
    'storage container': (187.0, 50.0, 115.0),
    'scale': (180.0, 141.0, 71.0),
    'tissue box': (77.0, 208.0, 35.0),
    'light switch': (72.0, 183.0, 168.0),
    'crate': (97.0, 99.0, 203.0),
    'power outlet': (172.0, 22.0, 158.0),
    'decoration': (155.0, 64.0, 40.0),
    'sign': (118.0, 159.0, 30.0),
    'projector': (69.0, 252.0, 148.0),
    'closet door': (45.0, 103.0, 173.0),
    'vacuum cleaner': (111.0, 38.0, 149.0),
    'candle': (184.0, 9.0, 49.0),
    'plunger': (188.0, 174.0, 67.0),
    'stuffed animal': (53.0, 206.0, 53.0),
    'headphones': (97.0, 235.0, 252.0),
    'dish rack': (66.0, 32.0, 182.0),
    'broom': (236.0, 114.0, 195.0),
    'guitar case': (241.0, 154.0, 83.0),
    'range hood': (133.0, 240.0, 52.0),
    'dustpan': (16.0, 205.0, 144.0),
    'hair dryer': (75.0, 101.0, 198.0),
    'water bottle': (237.0, 95.0, 251.0),
    'handicap bar': (191.0, 52.0, 49.0),
    'purse': (227.0, 254.0, 54.0),
    'vent': (49.0, 206.0, 87.0),
    'shower floor': (48.0, 113.0, 150.0),
    'water pitcher': (125.0, 73.0, 182.0),
    'mailbox': (229.0, 32.0, 114.0),
    'bowl': (158.0, 119.0, 28.0),
    'paper bag': (60.0, 205.0, 27.0),
    'alarm clock': (18.0, 215.0, 201.0),
    'music stand': (79.0, 76.0, 153.0),
    'projector screen': (134.0, 13.0, 116.0),
    'divider': (192.0, 97.0, 63.0),
    'laundry detergent': (108.0, 163.0, 18.0),
    'bathroom counter': (95.0, 220.0, 156.0),
    'object': (98.0, 141.0, 208.0),
    'bathroom vanity': (144.0, 19.0, 193.0),
    'closet wall': (166.0, 36.0, 57.0),
    'laundry hamper': (212.0, 202.0, 34.0),
    'bathroom stall door': (23.0, 206.0, 34.0),
    'ceiling light': (91.0, 211.0, 236.0),
    'trash bin': (79.0, 55.0, 137.0),
    'dumbbell': (182.0, 19.0, 117.0),
    'stair rail': (134.0, 76.0, 14.0),
    'tube': (87.0, 185.0, 28.0),
    'bathroom cabinet': (82.0, 224.0, 187.0),
    'cd case': (92.0, 110.0, 214.0),
    'closet rod': (168.0, 80.0, 171.0),
    'coffee kettle': (197.0, 63.0, 51.0),
    'structure': (175.0, 199.0, 77.0),
    'shower head': (62.0, 180.0, 98.0),
    'keyboard piano': (8.0, 91.0, 150.0),
    'case of water bottles': (77.0, 15.0, 130.0),
    'coat rack': (154.0, 65.0, 96.0),
    'storage organizer': (197.0, 152.0, 11.0),
    'folded chair': (59.0, 155.0, 45.0),
    'fire alarm': (12.0, 147.0, 145.0),
    'power strip': (54.0, 35.0, 219.0),
    'calendar': (210.0, 73.0, 181.0),
    'poster': (221.0, 124.0, 77.0),
    'potted plant': (149.0, 214.0, 66.0),
    'luggage': (72.0, 185.0, 134.0),
    'mattress': (42.0, 94.0, 198.0)
}




# Define PLY types
ply_dtypes = dict(
    [
        (b"int8", "i1"),
        (b"char", "i1"),
        (b"uint8", "u1"),
        (b"uchar", "u1"),
        (b"int16", "i2"),
        (b"short", "i2"),
        (b"uint16", "u2"),
        (b"ushort", "u2"),
        (b"int32", "i4"),
        (b"int", "i4"),
        (b"uint32", "u4"),
        (b"uint", "u4"),
        (b"float32", "f4"),
        (b"float", "f4"),
        (b"float64", "f8"),
        (b"double", "f8"),
    ]
)

# Numpy reader format
valid_formats = {"ascii": "", "binary_big_endian": ">", "binary_little_endian": "<"}


# ----------------------------------------------------------------------------------------------------------------------
#
#           Functions
#       \***************/
#


def parse_header(plyfile, ext):
    # Variables
    line = []
    properties = []
    num_points = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        if b"element" in line:
            line = line.split()
            num_points = int(line[2])

        elif b"property" in line:
            line = line.split()
            properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))

    return num_points, properties


def parse_mesh_header(plyfile, ext):
    # Variables
    line = []
    vertex_properties = []
    num_points = None
    num_faces = None
    current_element = None

    while b"end_header" not in line and line != b"":
        line = plyfile.readline()

        # Find point element
        if b"element vertex" in line:
            current_element = "vertex"
            line = line.split()
            num_points = int(line[2])

        elif b"element face" in line:
            current_element = "face"
            line = line.split()
            num_faces = int(line[2])

        elif b"property" in line:
            if current_element == "vertex":
                line = line.split()
                vertex_properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
            elif current_element == "vertex":
                if not line.startswith("property list uchar int"):
                    raise ValueError("Unsupported faces property : " + line)

    return num_points, num_faces, vertex_properties


def read_ply(filename, triangular_mesh=False):
    """
    Read ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to read.
    Returns
    -------
    result : array
        data stored in the file
    Examples
    --------
    Store data in file
    >>> points = np.random.rand(5, 3)
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example.ply', [points, values], ['x', 'y', 'z', 'values'])
    Read the file
    >>> data = read_ply('example.ply')
    >>> values = data['values']
    array([0, 0, 1, 1, 0])

    >>> points = np.vstack((data['x'], data['y'], data['z'])).T
    array([[ 0.466  0.595  0.324]
           [ 0.538  0.407  0.654]
           [ 0.850  0.018  0.988]
           [ 0.395  0.394  0.363]
           [ 0.873  0.996  0.092]])
    """

    with open(filename, "rb") as plyfile:

        # Check if the file start with ply
        if b"ply" not in plyfile.readline():
            raise ValueError("The file does not start whith the word ply")

        # get binary_little/big or ascii
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError("The file is not binary")

        # get extension for building the numpy dtypes
        ext = valid_formats[fmt]

        # PointCloud reader vs mesh reader
        if triangular_mesh:

            # Parse header
            num_points, num_faces, properties = parse_mesh_header(plyfile, ext)

            # Get point data
            vertex_data = np.fromfile(plyfile, dtype=properties, count=num_points)

            # Get face data
            face_properties = [
                ("k", ext + "u1"),
                ("v1", ext + "i4"),
                ("v2", ext + "i4"),
                ("v3", ext + "i4"),
            ]
            faces_data = np.fromfile(plyfile, dtype=face_properties, count=num_faces)

            # Return vertex data and concatenated faces
            faces = np.vstack((faces_data["v1"], faces_data["v2"], faces_data["v3"])).T
            data = [vertex_data, faces]

        else:

            # Parse header
            num_points, properties = parse_header(plyfile, ext)

            # Get data
            data = np.fromfile(plyfile, dtype=properties, count=num_points)

    return data


def header_properties(field_list, field_names):

    # List of lines to write
    lines = []

    # First line describing element vertex
    lines.append("element vertex %d" % field_list[0].shape[0])

    # Properties lines
    i = 0
    for fields in field_list:
        for field in fields.T:
            lines.append("property %s %s" % (field.dtype.name, field_names[i]))
            i += 1

    return lines


def write_ply(filename, field_list, field_names, triangular_faces=None):
    """
    Write ".ply" files
    Parameters
    ----------
    filename : string
        the name of the file to which the data is saved. A '.ply' extension will be appended to the
        file name if it does no already have one.
    field_list : list, tuple, numpy array
        the fields to be saved in the ply file. Either a numpy array, a list of numpy arrays or a
        tuple of numpy arrays. Each 1D numpy array and each column of 2D numpy arrays are considered
        as one field.
    field_names : list
        the name of each fields as a list of strings. Has to be the same length as the number of
        fields.
    Examples
    --------
    >>> points = np.random.rand(10, 3)
    >>> write_ply('example1.ply', points, ['x', 'y', 'z'])
    >>> values = np.random.randint(2, size=10)
    >>> write_ply('example2.ply', [points, values], ['x', 'y', 'z', 'values'])
    >>> colors = np.random.randint(255, size=(10,3), dtype=np.uint8)
    >>> field_names = ['x', 'y', 'z', 'red', 'green', 'blue', values']
    >>> write_ply('example3.ply', [points, colors, values], field_names)
    """

    # Format list input to the right form
    field_list = list(field_list) if (type(field_list) == list or type(field_list) == tuple) else list((field_list,))
    for i, field in enumerate(field_list):
        if field.ndim < 2:
            field_list[i] = field.reshape(-1, 1)
        if field.ndim > 2:
            print("fields have more than 2 dimensions")
            return False

    # check all fields have the same number of data
    n_points = [field.shape[0] for field in field_list]
    if not np.all(np.equal(n_points, n_points[0])):
        print("wrong field dimensions")
        return False

    # Check if field_names and field_list have same nb of column
    n_fields = np.sum([field.shape[1] for field in field_list])
    if n_fields != len(field_names):
        print("wrong number of field names")
        return False

    # Add extension if not there
    if not filename.endswith(".ply"):
        filename += ".ply"

    # open in text mode to write the header
    with open(filename, "w") as plyfile:

        # First magical word
        header = ["ply"]

        # Encoding format
        header.append("format binary_" + sys.byteorder + "_endian 1.0")

        # Points properties description
        header.extend(header_properties(field_list, field_names))

        # Add faces if needded
        if triangular_faces is not None:
            header.append("element face {:d}".format(triangular_faces.shape[0]))
            header.append("property list uchar int vertex_indices")

        # End of header
        header.append("end_header")

        # Write all lines
        for line in header:
            plyfile.write("%s\n" % line)

    # open in binary/append to use tofile
    with open(filename, "ab") as plyfile:

        # Create a structured array
        i = 0
        type_list = []
        for fields in field_list:
            for field in fields.T:
                type_list += [(field_names[i], field.dtype.str)]
                i += 1
        data = np.empty(field_list[0].shape[0], dtype=type_list)
        i = 0
        for fields in field_list:
            for field in fields.T:
                data[field_names[i]] = field
                i += 1

        data.tofile(plyfile)

        if triangular_faces is not None:
            triangular_faces = triangular_faces.astype(np.int32)
            type_list = [("k", "uint8")] + [(str(ind), "int32") for ind in range(3)]
            data = np.empty(triangular_faces.shape[0], dtype=type_list)
            data["k"] = np.full((triangular_faces.shape[0],), 3, dtype=np.uint8)
            data["0"] = triangular_faces[:, 0]
            data["1"] = triangular_faces[:, 1]
            data["2"] = triangular_faces[:, 2]
            data.tofile(plyfile)

    return True


def describe_element(name, df):
    """Takes the columns of the dataframe and builds a ply-like description
    Parameters
    ----------
    name: str
    df: pandas DataFrame
    Returns
    -------
    element: list[str]
    """
    property_formats = {"f": "float", "u": "uchar", "i": "int"}
    element = ["element " + name + " " + str(len(df))]

    if name == "face":
        element.append("property list uchar int points_indices")

    else:
        for i in range(len(df.columns)):
            # get first letter of dtype to infer format
            f = property_formats[str(df.dtypes[i])[0]]
            element.append("property " + f + " " + df.columns.values[i])

    return element