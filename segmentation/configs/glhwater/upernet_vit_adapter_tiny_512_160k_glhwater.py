# Copyright (c) Shanghai AI Lab. All rights reserved.
_base_ = [
    '../_base_/datasets/siri_224.py',
    '../_base_/schedules/imagenet_bs1024_adamw_swin.py',
    '../_base_/default_runtime.py'
]

# model settings
model = dict(
    type='ImageClassifier',
    backbone=dict(
        type='VisionTransformer',
        img_size=512,
        arch={"embed_dims": 192,
              "num_layers": 12,
              "num_heads": 4,
              "feedforward_channels": 768}, ),
    neck=None,
    head=dict(
        type='VisionTransformerClsHead',
        num_classes=12,
        in_channels=192,
        loss=dict(
            type='LabelSmoothLoss', label_smooth_val=0.1,
            mode='classy_vision'),
    ),
    init_cfg=[
        dict(type='TruncNormal', layer='Linear', std=0.02, bias=0.),
        dict(type='Constant', layer='LayerNorm', val=1., bias=0.)
    ],
    train_cfg=dict()
)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
checkpoint_config = dict(
    interval=1,
    max_keep_ckpts=2,
    save_last=True)
evaluation = dict(interval=10)
optimizer = dict(lr=5e-4)
# 8 gpus
data = dict(
    samples_per_gpu=64,
)
