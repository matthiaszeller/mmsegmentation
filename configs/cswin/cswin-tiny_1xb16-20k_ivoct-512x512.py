_base_ = [
    './upernet_cswin_tiny.py', '../_base_/datasets/ivoct.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
)


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

warmup_iters = 180
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=warmup_iters,
        end=20000,
        by_epoch=False,
    )
]

train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader

default_hooks = dict(
    # Adjust logging interval
    logger=dict(interval=20),
    # Checkpointing
    checkpoint=dict(save_best='metric/mIoU', rule='greater', max_keep_ckpts=10),
)
# Evaluate more often, takes few seconds only in this config
train_cfg = dict(val_interval=200)
