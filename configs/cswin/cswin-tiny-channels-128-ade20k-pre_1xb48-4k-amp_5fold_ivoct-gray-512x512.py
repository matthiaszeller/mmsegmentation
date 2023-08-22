_base_ = [
    './upernet_cswin_tiny.py', '../_base_/datasets/ivoct_polar_gray.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=None),
    decode_head=dict(
        channels=128
    ),
    auxiliary_head=dict(
        channels=128
    )
)

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006 * 4, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))


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

default_hooks = dict(
    # Adjust logging interval
    logger=dict(interval=20),
    # Checkpointing
    checkpoint=dict(save_best='metric/mIoU', rule='greater', max_keep_ckpts=10),
)

# 4k iterations
train_cfg = dict(max_iters=4000, val_interval=100)


train_dataloader=dict(
    batch_size=48,
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
