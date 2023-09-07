_base_ = [
    '../_base_/models/upernet_hivit_tiny.py', '../_base_/datasets/ivoct_polar_gray.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        channels=128,
        num_classes=2,
    ),
    auxiliary_head=dict(
        channels=128,
        num_classes=2
    )
)

train_dataloader=dict(
    batch_size=40,
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
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.)
        }))


# 4k iterations
warmup_iters = 45
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=1e-8,
        power=1.0,
        begin=warmup_iters,
        end=4000,
        by_epoch=False,
    )
]

# Train config
default_hooks = dict(
    # Adjust logging interval
    logger=dict(interval=20),
    # Checkpointing
    checkpoint=dict(save_best='metric/mIoU', rule='greater', max_keep_ckpts=10),
)
# Evaluate more often, takes few seconds only in this config
train_cfg = dict(max_iters=4000, val_interval=200)
