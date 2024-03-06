_base_ = ["../_base_/default_runtime.py"]
# misc custom setting

data_processing = "subcloud" # partition or subcloud

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

batch_size = 6  # bs: total bs in all gpus
mix_prob = 0
empty_cache = False
enable_amp = True

# model settings
groupers = [
    dict(type='ball-grouper', max_radius=0.4, min_radius=0, nsample=32),
    dict(type='ball-grouper', max_radius=0.8, min_radius=0, nsample=32),
    dict(type='ball-grouper', max_radius=1.6, min_radius=0, nsample=32),
    dict(type='ball-grouper', max_radius=3.2, min_radius=0, nsample=32),
    dict(type='ball-grouper', max_radius=6.4, min_radius=0, nsample=32),
]
enc_blocks = [
    dict(type="KPConvSimpleBlock", in_channels=32, KP_extent=0.2, repeats=1, grouper=groupers[0]),
    dict(type="KPConvResnetBBlock", in_channels=64, KP_extent=0.4, repeats=6, grouper=groupers[1]),
    dict(type="KPConvResnetBBlock", in_channels=128, KP_extent=0.8, repeats=14, grouper=groupers[2]),
    dict(type="KPConvResnetBBlock", in_channels=256, KP_extent=1.6, repeats=18, grouper=groupers[3]),
    dict(type="KPConvResnetBBlock", in_channels=512, KP_extent=3.2, repeats=12, grouper=groupers[4]),
]
dec_blocks = [
    None, None, None, None
]
samplers = [
    dict(type="rs", stride=4),
    dict(type="rs", stride=4),
    dict(type="rs", stride=4),
    dict(type="rs", stride=4),
]
poolings = [
    dict(type='knn-pool', nsample=16, dilation=0, reduction="max"),
    dict(type='knn-pool', nsample=16, dilation=0, reduction="max"),
    dict(type='knn-pool', nsample=16, dilation=0, reduction="max"),
    dict(type='knn-pool', nsample=16, dilation=0, reduction="max"),
]


model = dict(
    type="BaseSemanticSegmentor",
    encoder=dict(
        type="LDCEncoder",
        in_channels=4,
        enc_blocks=enc_blocks,
        g_blocknumber=[1, 1, 1, 1, 1],
        samplers=samplers, poolings=poolings,
        rgb_importance=2,
        channels=[32, 64, 128, 256, 512],
        down=[True, True, True, True]
    ),
    decoder=dict(
        type="BaseSegDecoder",
        dec_blocks=dec_blocks,
    ),
    head=dict(
        type="OffsetSemSegHead",
        num_classes=19, 
        in_channels=32,
    ),
    criteria=[dict(type="CrossEntropyLoss", weight=[
                3.1557,
                8.7029,
                7.8281,
                6.1354,
                6.3161,
                7.9937,
                8.9704,
                10.1922,
                1.6155,
                4.2187,
                1.9385,
                5.5455,
                2.0198,
                2.6261,
                1.3212,
                5.1102,
                2.5492,
                5.8585,
                7.3929,
            ], loss_weight=1.0, ignore_index=-1)],
)

# scheduler settings
epoch = 500
eval_epoch = 100
optimizer = dict(type="AdamW", lr=0.002, weight_decay=0.005)
scheduler = dict(
    type="OneCycleLR",
    max_lr=optimizer["lr"],
    pct_start=0.04,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=100.0,
)

grid_size = 0.2

# dataset settings
if data_processing == "partition":
    dataset_type = "SemanticKittiPartitionDataset"
    kwargs = dict(partition_size=grid_size, preload_data_to_memory=False, preload_partition_info=True)
elif data_processing == "subcloud":
    dataset_type = "SemanticKittiSubCloudDataset"
    kwargs = dict(voxel_size=grid_size, preload_subcloud_info=True)
else:
    raise NotImplementedError

#data_root = data_root = "/home/magic/magic/Datasets/UnifiedFormat/semantickitti"
data_root = "./data/semantickitti"
input_points = 100000
data = dict(
    num_classes=19,
    ignore_index=-1,
    train=dict(
        **kwargs,
        type=dataset_type,
        split="train",
        data_root=data_root,
        transform=[
            dict(type="SphereCrop", point_max=input_points, mode='random'),
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomRotate", angle=[-1.57, 1.57], axis='z', center=[0, 0, 0], p=0.5),
            dict(type="RandomScale", scale=[0.95, 1.05]),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                feat_keys=("coord", "strength"),
            ),
        ],
        test_mode=False
    ),
    val=dict(
        **kwargs,
        type=dataset_type,
        loop=1,
        split="val",
        data_root=data_root,
        transform=[
            dict(type="SphereCrop", point_max=input_points, mode='random'),
            dict(type="CenterShift", apply_z=True),
            dict(type="ToTensor"),
            dict(
                type="Collect",
                keys=("coord", "segment"),
                offset_keys_dict=dict(offset="coord"),
                feat_keys=["coord", "strength"],
            )
        ],
        test_mode=False),
    test=dict(
        **kwargs,
        type=dataset_type,
        split="val",
        data_root=data_root,
        transform=[
        ],
        test_mode=True,
        test_cfg=dict(
            crop=dict(type="SphereCrop", point_max=input_points, mode='all'),
            post_transform=[
                dict(type='CenterShift', apply_z=True),
                dict(type='ToTensor'),
                dict(
                    type='Collect',
                    keys=('coord', 'index'),
                    feat_keys=('coord', 'strength'))
            ],
            aug_transform=[  
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    )
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[0.95, 0.95]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[0],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[1],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ],
                [
                    dict(
                        type="RandomRotateTargetAngle",
                        angle=[3 / 2],
                        axis="z",
                        center=[0, 0, 0],
                        p=1,
                    ),
                    dict(type="RandomScale", scale=[1.05, 1.05]),
                ], 
                [dict(type="RandomFlipV2", p=0)],
                [dict(type="RandomFlipV2", p=1, axis=[0])],
                [dict(type="RandomFlipV2", p=1, axis=[1])],
                [dict(type="RandomFlipV2", p=1, axis=[0, 1])],
            ]
        )
    )
)
