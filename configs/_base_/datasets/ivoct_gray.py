
_base_ = [
    './ivoct.py'
]

crop_size = (512, 512)

# images are PNG in palette mode, unchanged and pillow args will only load grayscale values
_load_img = dict(type='LoadImageFromZipFile', imdecode_backend='pillow', color_type='unchanged')

train_pipeline = [
    _load_img,
    dict(type='DuplicateImageChannels', num_repeat=3),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    dict(type='RandomRotate', prob=1., degree=360, seg_pad_val=0),
    #dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    _load_img,
    dict(type='DuplicateImageChannels', num_repeat=3),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


train_dataloader = dict(
    dataset=dict(
        pipeline=train_pipeline
    )
)

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)

test_dataloader = val_dataloader
