_base_ = [
    '../_base_/models/upernet_cswin.py'
]

data_preprocessor = dict(
    # green channel statistics
    mean=[116.28],
    std=[57.12],
)

model = dict(
    type='EncoderDecoderPseudo3D',
    backbone=dict(
        init_cfg=None,
        in_channels=4,
        embed_dims=64,
        depths=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        drop_path_rate=0.3,
        use_cp=False,
    ),
    decode_head=dict(
        type='UPerHeadPseudo3D',
        in_channels=[64, 128, 256, 512],
        num_classes=2
    ),
    auxiliary_head=dict(
        type='FCNHeadPseudo3D',
        in_channels=256,
        num_classes=2
    ))
