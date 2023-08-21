_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ade20k_640x640.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_80k.py'
]
crop_size = (640, 640)
data_preprocessor = dict(size=crop_size)

norm_cfg = dict(type='SyncBN', requires_grad=True)
pretrained = '../mmsegmentation/checkpoints/mae_hivit2_base_1600ep.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='HiViT2',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        img_size=224,
        task_img_size=640,
        patch_size=16,
        embed_dim=512,
        depths=[2, 2, 20],
        num_heads=8,
        mlp_ratio=4.,
        rpe=True,
        drop_path_rate=0.1,
        with_fpn=True,
        out_indices=['H', 'M', 19, 19],
        use_checkpoint=False,
    ),
    decode_head=dict(
        in_channels=[128, 256, 512, 512],
        num_classes=150,
        channels=512,
    ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=256,
        in_index=1,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4
        )
    ),
    test_cfg = dict(mode='slide', crop_size=crop_size, stride=(426, 426))
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        # no weight decay for biases
        bias_decay_mult=0.0,
        # no weight decay for pos embeddings and norm layers
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=80000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=6)
train_cfg = dict(val_interval=500)
default_hooks = dict(
    logger=dict(interval=50)
)

# img_norm_cfg = dict(
#     mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (640, 640)
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=(2560, 640),
#         img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
#         flip=True,
#         transforms=[
#             dict(type='SETR_Resize', keep_ratio=True,
#                  crop_size=crop_size, setr_multi_scale=True),
#             dict(type='RandomFlip'),
#             dict(type='Normalize', **img_norm_cfg),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='Collect', keys=['img']),
#         ])
# ]
#
# # By default, models are trained on 8 GPUs with 2 images per GPU
# val_dataloader = dict(dataset=dict(pipeline=test_pipeline))
# test_dataloader = val_dataloader
