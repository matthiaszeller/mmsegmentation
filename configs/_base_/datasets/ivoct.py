# dataset settings
dataset_type = 'IVOCTZipDataset'
data_root = 'data/shockwave'

crop_size = (512, 512)

train_pipeline = [
    # images are PNG in palette mode, by default mmcv.imgfrombytes will apply colormap
    dict(type='LoadImageFromZipFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    #dict(type='RandomResize', scale=(512, 512), ratio_range=(0.5, 2.0), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    #dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

test_pipeline = [
    # images are PNG in palette mode, by default mmcv.imgfrombytes will apply colormap
    dict(type='LoadImageFromZipFile'),
    dict(type='Resize', scale=(512, 512), keep_ratio=True),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]


# for Test Time Augmentation (tools/test.py --tta)
tta_pipeline = None

train_dataloader = dict(
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images-3D-cartesian', seg_map_path='labels/cartesian'),
        ann_file='splits/segmentation/train.txt',
        pipeline=train_pipeline,
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(img_path='images-3D-cartesian', seg_map_path='labels/cartesian'),
        ann_file='splits/segmentation/val.txt',
        pipeline=test_pipeline,
    )
)

test_dataloader = val_dataloader

val_evaluator = dict(type='ClassIoUMetric', iou_metrics=['mIoU', 'mFscore', 'mDice'], prefix='metric')
test_evaluator = val_evaluator
