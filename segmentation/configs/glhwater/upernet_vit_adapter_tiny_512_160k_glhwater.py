# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/models/upernet_r50.py',
    '../_base_/datasets/glhwater.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_160k.py'
]

pretrained = None

# model settings
model = dict(
    pretrained=pretrained,
    backbone=dict(
        _delete_=True,
        type='ViT_Adapter',
        img_size=512,
        pretrain_size=512,
        depth=12,
        embed_dim=192,
        num_heads=4,
        mlp_ratio=4,
        # adapter param
        drop_path_rate=0.1,
        conv_inplane=64,
        n_points=4,
        deform_num_heads=6,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
    ),
    decode_head=dict(num_classes=2, in_channels=[192, 192, 192, 192],
                     loss_decode=dict(class_weight=[1, 2],)),
    auxiliary_head=dict(num_classes=2, in_channels=192,
                        loss_decode=dict(class_weight=[1, 2],)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(341, 341))
)
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 512),
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='ResizeToMultiple', size_divisor=32),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
optimizer = dict(_delete_=True, type='AdamW', lr=6e-4, betas=(0.9, 0.999), weight_decay=0.01,
                 #  constructor='MyLayerDecayOptimizerConstructor',
                 paramwise_cfg=dict(num_layers=12, layer_decay_rate=0.95))
lr_config = dict(_delete_=True, policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)
log_config = dict(interval=10)
# 8 gpus
data = dict(samples_per_gpu=16,
            workers_per_gpu=12,
            val=dict(pipeline=test_pipeline),
            test=dict(pipeline=test_pipeline))
runner = dict(type='IterBasedRunner', max_iters=30000)
checkpoint_config = dict(by_epoch=False, interval=1000, max_keep_ckpts=1)
evaluation = dict(interval=2000, metric='mIoU', save_best='mIoU')
fp16 = dict(loss_scale=dict(init_scale=512))
