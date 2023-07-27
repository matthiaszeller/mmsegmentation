_base_ = [
    '../_base_/models/upernet_cswin.py'
]

model = dict(
    backbone=dict(
        init_cfg=None,
        embed_dims=64,
        depths=[1, 2, 21, 1],
        num_heads=[2, 4, 8, 16],
        split_size=[1, 2, 7, 7],
        drop_path_rate=0.3,
        use_cp=False,
    ),
    decode_head=dict(
        in_channels=[64,128,256,512],
        num_classes=2
    ),
    auxiliary_head=dict(
        in_channels=256,
        num_classes=2
    ))
