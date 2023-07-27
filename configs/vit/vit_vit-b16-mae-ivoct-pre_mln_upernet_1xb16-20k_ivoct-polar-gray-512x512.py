_base_ = [
    '../_base_/models/upernet_vit-b16_ln_mln.py',
    '../_base_/datasets/ivoct_polar_gray_1chan.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[116.28],
    std=[57.12],
    size=crop_size,
)


model = dict(
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        in_channels=1,
        # modify model.backbone.init_cfg.checkpoint with script arguments
        init_cfg=dict(type='Pretrained', checkpoint=None),
    ),
    decode_head=dict(num_classes=2),
    auxiliary_head=dict(num_classes=2)
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006 * 16 / 2, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'pos_embed': dict(decay_mult=0.),
            'cls_token': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=370),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=370,
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
    checkpoint=dict(save_best='metric/mIoU.calcium', rule='greater', max_keep_ckpts=10),
)
# Evaluate more often, takes few seconds only in this config
train_cfg = dict(val_interval=200)
