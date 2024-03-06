_base_ = ["../_base_/default_runtime.py"]

data_processing = "partition" # partition or subcloud


groupers = [
    dict(type='knn-grouper', nsample=8, dilation=0),
    dict(type='knn-grouper', nsample=16, dilation=0),
    dict(type='knn-grouper', nsample=16, dilation=0),
    dict(type='knn-grouper', nsample=16, dilation=0),
    dict(type='knn-grouper', nsample=16, dilation=0),
]
enc_blocks = [
    dict(type="PointTransformerBlock", in_channels=32, repeats=1, grouper=groupers[0]),
    dict(type="PointTransformerBlock", in_channels=64, repeats=2, grouper=groupers[1]),
    dict(type="PointTransformerBlock", in_channels=128, repeats=3, grouper=groupers[2]),
    dict(type="PointTransformerBlock", in_channels=256, repeats=5, grouper=groupers[3]),
    dict(type="PointTransformerBlock", in_channels=512, repeats=2, grouper=groupers[4]),
]
dec_blocks = [
    dict(type="PointTransformerBlock", in_channels=512, repeats=1, grouper=groupers[4]),
    dict(type="PointTransformerBlock", in_channels=256, repeats=1, grouper=groupers[3]),
    dict(type="PointTransformerBlock", in_channels=128, repeats=1, grouper=groupers[2]),
    dict(type="PointTransformerBlock", in_channels=64, repeats=1, grouper=groupers[1]),
    dict(type="PointTransformerBlock", in_channels=32, repeats=1, grouper=groupers[0]),
]
samplers = [
    dict(type="fps", stride=4),
    dict(type="fps", stride=4),
    dict(type="fps", stride=4),
    dict(type="fps", stride=4),
]
poolings = [
    dict(type='grid-pool', grid_size=0.08, reduction="max"),
    dict(type='grid-pool', grid_size=0.16, reduction="max"),
    dict(type='grid-pool', grid_size=0.32, reduction="max"),
    dict(type='grid-pool', grid_size=0.64, reduction="max"),
]


model = dict(
    type="BaseSemanticSegmentor",
    encoder=dict(
        type="LDCEncoder",
        in_channels=6,
        enc_blocks=enc_blocks,
        g_blocknumber=[1, 1, 1, 1, 1],
        samplers=samplers, poolings=poolings,
        channels=[32, 64, 128, 256, 512],
        down=[True, True, True, True]
    ),
    decoder=dict(
        type="BaseSegDecoder",
        dec_blocks=dec_blocks,
    ),
    head=dict(
        type="OffsetSemSegHead",
        num_classes=13, 
        in_channels=32,
    ),
    criteria=[dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1)],
)

if data_processing == "partition":
    test = dict(
        type="SemSegPartitionTester"
    )
elif data_processing == "subcloud":
    test = dict(
        type="SemSegSubCloudTester"
    )
else:
    raise NotImplementedError

# misc custom setting
batch_size = 8  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True

# scheduler settings
epoch = 900
optimizer = dict(type="AdamW", lr=0.006, weight_decay=0.05)
#scheduler = dict(type='MultiStepLR', milestones=[0.6, 0.8], gamma=0.1)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

# dataset settings
grid_size = 0.04

# dataset settings
if data_processing == "partition":
    dataset_type = "S3DISPartitionDataset"
    kwargs = dict(partition_size=grid_size, preload_data_to_memory=False, preload_partition_info=False)
elif data_processing == "subcloud":
    dataset_type = "S3DISSubCloudDataset"
    kwargs = dict(voxel_size=grid_size, preload_subcloud_info=True)
else:
    raise NotImplementedError

data_root = "./data/s3dis"
input_points = 100000
split_file="Area1"
data = dict(
    num_classes=13,
    ignore_index=-1,
    train=dict(
        **kwargs,
        type=dataset_type,
        split="train",
        data_root=data_root,
        split_file=split_file,
        transform=[
            dict(type="SphereCrop", point_max=input_points, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            #dict(type="RandomColorDrop", p=0.2, color_augment=0.0),
            dict(type="HueSaturationTranslation", hue_max=0.2, saturation_max=0.2),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    val=dict(
        **kwargs,
        type=dataset_type,
        split="val",
        split_file=split_file,
        data_root=data_root,
        transform=[
            dict(type="SphereCrop", point_max=input_points, mode="random"),
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "color"),
            ),
        ],
        test_mode=False,
    ),
    test=dict(
        **kwargs,
        type=dataset_type,
        split="val",
        split_file=split_file,
        data_root=data_root,
        transform=[
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            crop=dict(type="SphereCrop", point_max=input_points, mode="all"),
            post_transform=[
                dict(type="CenterShift", apply_z=True),
                dict(type="ToTensor"),
                dict(
                    type="Collect",
                    keys=("coord", "index"),
                    feat_keys=("coord", "color"),
                ),
            ],
            aug_transform=[
                [dict(type="RandomScale", scale=[0.9, 0.9])],
                [dict(type="RandomScale", scale=[0.95, 0.95])],
                [dict(type="RandomScale", scale=[1, 1])],
                [dict(type="RandomScale", scale=[1.05, 1.05])],
                [dict(type="RandomScale", scale=[1.1, 1.1])],
                [
                    dict(type="RandomScale", scale=[0.9, 0.9]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                    dict(type="RandomFlip", p=1),
                ],
                [dict(type="RandomScale", scale=[1, 1]), dict(type="RandomFlip", p=1)],
                [
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                    dict(type="RandomFlip", p=1),
                ],
                [
                    dict(type="RandomScale", scale=[1.1, 1.1]),
                    dict(type="RandomFlip", p=1),
                ],
            ],
        ),
    ),
)
