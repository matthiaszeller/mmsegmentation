_base_ = [
    './cswin-tiny_1xb16-20k_ivoct-512x512.py'
]

crop_size = (512, 512)
train_pipeline = [
    # note: loading relies on mmcv.frombytes, which has BGR and not RGB
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='RandomResize', scale=(512, 512), ratio_range=(0.667, 1.5), keep_ratio=True),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)
