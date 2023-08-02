_base_ = [
    './cswin-tiny-ade20k-pre_1xb16-20k_ivoct-polar-gray-512x512.py'
]

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None),
        decode_head=dict(
        channels=256,
        in_index=[1,2,3],
        in_channels=[128,256,512]
    )
)

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


optim_wrapper = dict(
    optimizer=dict(lr=0.00006 * 4),
)

# 4k iterations
train_cfg = dict(max_iters=4000)
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
