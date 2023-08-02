_base_ = [
    './ivoct_polar.py'
]

crop_size = (512, 512)

# images are PNG in palette mode, unchanged and pillow args will only load grayscale values
_load_img = dict(type='LoadImageFromZipFile', imdecode_backend='pillow', color_type='unchanged', enable_3d=True)

train_pipeline = [
    _load_img,
    dict(type='LoadAnnotations', enable_3d=True),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    #dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5, direction='vertical'),
    dict(type='RandomRoll', axis=0),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs', enable_3d=True)
]

test_pipeline = [
    _load_img,
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', enable_3d=True),
    dict(type='PackSegInputs', enable_3d=True)
]


train_dataloader = dict(
    dataset=dict(
        enable_3d=True,
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    dataset=dict(
        enable_3d=True,
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader
