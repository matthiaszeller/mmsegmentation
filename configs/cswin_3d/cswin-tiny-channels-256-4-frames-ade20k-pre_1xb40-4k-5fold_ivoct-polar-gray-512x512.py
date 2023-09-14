_base_ = [
    './upernet_cswin_tiny_3D.py', '../_base_/datasets/ivoct_polar_gray_3D.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(
    # green channel statistics
    mean=[116.28],
    std=[57.12],
    #
    size=crop_size
)

model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=None),
    decode_head=dict(
        channels=256
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
            'norm': dict(decay_mult=0.)
        }))

warmup_iters = 180
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=1e-7,
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
# Evaluate more often, takes few seconds only in this config
train_cfg = dict(val_interval=200, max_iters=4000)


