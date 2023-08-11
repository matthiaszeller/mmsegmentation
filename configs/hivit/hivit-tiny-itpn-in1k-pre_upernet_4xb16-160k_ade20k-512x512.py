_base_ = [
    '../_base_/models/upernet_hivit_tiny.py', '../_base_/datasets/ade20k-size16.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_160k.py'
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        # change with --cfg-options model.backbone.init_cfg.checkpoint=<path to checkpoint>
        init_cfg=dict(type='Pretrained', checkpoint=None),
        pretrain_img_size=224,
        img_size=512,
    ),
    decode_head=dict(num_classes=150),
    auxiliary_head=dict(num_classes=150)
)

# optimizer wrapper
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    loss_scale='dynamic',
    optimizer=dict(
        type='AdamW',
        lr=1.5e-4 * 4096 / 256,
        betas=(0.9, 0.95),
        weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'norm': dict(decay_mult=0.0),
            'bias': dict(decay_mult=0.0),
            'pos_embed': dict(decay_mult=0.),
            'mask_token': dict(decay_mult=0.),
        }))

param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=1500,
        end=160_000,
        by_epoch=False,
    )
]

# By default, models are trained on 8 GPUs with 2 images per GPU
train_dataloader = dict(batch_size=16)
val_dataloader = dict(batch_size=1)
test_dataloader = val_dataloader
