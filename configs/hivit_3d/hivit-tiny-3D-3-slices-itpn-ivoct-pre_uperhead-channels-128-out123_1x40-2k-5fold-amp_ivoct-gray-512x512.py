_base_ = [
    '../_base_/models/upernet_hivit_tiny.py', '../_base_/datasets/ivoct_polar_gray_3D.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(
    size=crop_size,
    mean=[24.87],
    std=[53.36],
)

checkpoint = 'checkpoints/itpn-pixel_3D-3-slices_hivit-tiny-p16-ape-alpha_4xb30-amp-coslr-400e_ivoct-gray_epoch-275.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='HiViT3D',
        n_slice=3,
        slices_to_batch=True,
        in_chans=1,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint),
        pretrain_img_size=512,
        img_size=512,
        out_indices=(0, 1, 2, ),
        ape=True,
        rpe=False,
    ),
    decode_head=dict(
        type='UPerHead3D',
        n_slice=3,
        channels=128,
        num_classes=2,
        in_channels=[96, 192, 384, ],
        in_index=(0, 1, 2, ),
    ),
    auxiliary_head=None
)

train_dataloader = dict(
    batch_size=16,
    dataset=dict(
        # use --cfg-options to replace this argument with the correct fold id
        ann_file='splits/segmentation/train_fold<i>.txt'
    )
)

val_dataloader=dict(
    dataset=dict(
        # use --cfg-options to replace this argument with the correct fold id
        ann_file='splits/segmentation/val_fold<i>.txt'
    )
)

test_dataloader=val_dataloader

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006 * 2, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        # no weight decay for biases
        bias_decay_mult=0.0,
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
        }
    )
)


# 4k iterations
warmup_iters = 45
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=0,
        power=1.0,
        begin=warmup_iters,
        end=2000,
        by_epoch=False,
    )
]

# Train config
default_hooks = dict(
    # Adjust logging interval
    logger=dict(interval=20),
    # Checkpointing
    checkpoint=dict(save_best='metric/mIoU', rule='greater', interval=2000),
)
# Evaluate more often, takes few seconds only in this config
train_cfg = dict(max_iters=2000, val_interval=200)
