_base_ = [
    '../_base_/models/upernet_cswin.py', '../_base_/datasets/ivoct.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        init_cfg=None,
        embed_dims=64,
        depths=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        drop_path_rate=0.3,
        use_cp=False,
    ),
    decode_head=dict(
        in_channels=[64,128,256,512],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=2
    ))

# AdamW optimizer, no weight decay for position embedding & layer norm in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
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

# Adjust logging interval
default_hooks = dict(
    logger=dict(interval=10),
)
