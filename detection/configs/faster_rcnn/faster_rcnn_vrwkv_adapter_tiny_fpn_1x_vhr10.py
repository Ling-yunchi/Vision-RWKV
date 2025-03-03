# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py',
    '../_base_/datasets/vhr10.py',
    '../_base_/schedules/schedule_1x.py',
    '../_base_/default_runtime.py'
]

pretrained = None
# https://huggingface.co/OpenGVLab/Vision-RWKV/resolve/main/vrwkv_t_in1k_224.pth

model = dict(
    backbone=dict(
        _delete_=True,
        type='VRWKV_Adapter',
        img_size=224,
        patch_size=16,
        embed_dims=192,
        depth=12,
        pretrained=pretrained,
        init_values=1.,
        with_cp=False,
        # adapter param
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    ),
    neck=dict(
        type='FPN',
        in_channels=[192, 192, 192, 192],
        out_channels=256,
        num_outs=5),
    roi_head=dict(
        bbox_head=dict(num_classes=10,),
    ))
# optimizer

# 8 gpus
data = dict(samples_per_gpu=4,
            workers_per_gpu=4)

optimizer = dict(
    _delete_=True, type='AdamW', lr=0.0002, weight_decay=0.01,
    # constructor='LayerDecayOptimizerConstructor',
    paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.85))
optimizer_config = dict(grad_clip=None)

checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True,
)
log_config = dict(interval=10)
