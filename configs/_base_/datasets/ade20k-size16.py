# dataset settings
_base_ = ['./ade20k.py']


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='ResizeToMultiple', size_divisor=(16, 16)),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline
    )
)
test_dataloader = val_dataloader
