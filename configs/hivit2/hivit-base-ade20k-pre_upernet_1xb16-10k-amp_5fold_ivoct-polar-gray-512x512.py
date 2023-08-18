_base_ = [
    './hivit-base-ade20k-pre_upernet_1xb16-10k-amp_ivoct-polar-gray-512x512.py'
]

train_dataloader = dict(
    # use --cfg-options to replace this argument with the correct fold id
    dataset=dict(
        ann_file='splits/segmentation/train_fold<i>.txt'
    )
)

val_dataloader = dict(
    # use --cfg-options to replace this argument with the correct fold id
    dataset=dict(
        ann_file='splits/segmentation/train_fold<i>.txt'
    )
)
