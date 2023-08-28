# model settings
backbone_norm_cfg = dict(type='LN', requires_grad=True)
norm_cfg = dict(type='SyncBN', requires_grad=True)

data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='HiViT',
        arch='tiny',
        ape=True,
        rpe=True,
        drop_path_rate=0.1,
        norm_cfg=backbone_norm_cfg,
        format_output=False,
    ),
    neck=dict(
        type='iTPNNeck',
        embed_dim=384,
        mlp_ratio=4.,
        #  transformer pyramid
        fpn_dim=256,
        fpn_depth=2,
        num_outs=4,
    ),
    decode_head=dict(
        type='UPerHead',
        in_channels=[352, 448, 640, 640],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=64,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=None,
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)  # yapf: disable
