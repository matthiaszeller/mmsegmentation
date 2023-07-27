_base_ = [
    './cswin-tiny-ade20k-pre_1xb16-20k_ivoct-polar-gray-512x512.py'
]

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None),
        decode_head=dict(
        channels=256
    )
)

train_dataloader=dict(
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


# 10k iterations
train_cfg = dict(max_iters=10000)
warmup_iters = 180
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=warmup_iters),
    dict(
        type='PolyLR',
        eta_min=1e-8,
        power=1.0,
        begin=warmup_iters,
        end=10000,
        by_epoch=False,
    )
]
