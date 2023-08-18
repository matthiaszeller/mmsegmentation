_base_ = [
    '../_base_/models/upernet_swin.py', '../_base_/datasets/ivoct_polar_gray.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

pretrained = '../mmsegmentation/checkpoints/mae_hivit2_base_1600ep_ft100ep.pth'
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        _delete_=True,
        type='HiViT2',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained),
        img_size=640,
        task_img_size=512,
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
        num_classes=2,
        channels=128,
    ),
    auxiliary_head=None,
    # auxiliary_head=dict(
    #     in_channels=512,
    #     num_classes=150
    # ),
    test_cfg=dict(mode='whole')
)

# AdamW optimizer, no weight decay for position embedding & layer norm
# in backbone
optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.),
            'bias': dict(decay_mult=0.)
        }))


param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=120),
    dict(
        type='PolyLR',
        eta_min=0.0,
        power=1.0,
        begin=120,
        end=10000,
        by_epoch=False,
    )
]

train_cfg = dict(max_iters=10000, val_interval=200)

train_dataloader = dict(batch_size=16)
