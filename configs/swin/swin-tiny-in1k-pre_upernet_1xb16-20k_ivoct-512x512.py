_base_ = [
    './swin-tiny_1xb16-20k_ivoct-512x512.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa
model = dict(
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Pretrained', checkpoint=checkpoint_file),
    ),
)
