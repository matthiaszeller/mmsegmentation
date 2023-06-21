# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='CSWinTransformer2',
        embed_dim=64,
        patch_size=4,
        depth=[1, 2, 21, 1],
        num_heads=[2,4,8,16],
        split_size=[1,2,7,7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.1),
    decode_head=dict(
        type='UPerHead',
        in_channels=[96, 192, 384, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='LovaszLoss', loss_weight=1.0, per_image=True)),
#             type='LovaszLoss', loss_weight=1.0, reduction='none')),
#             type='DiceLoss', use_sigmoid=False, loss_weight=1.0)),
#             type='FocalLoss', loss_weight=1.0)),
#             type='CrossEntropyLoss', loss_weight=1.0)),
#         loss_decode=[dict(type='CrossEntropyLoss', loss_name='loss_ce', loss_weight=1.0),
#             dict(type='LovaszLoss', per_image=True, loss_weight=3.0)]),
    
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=384,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=True,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='LovaszLoss', loss_weight=0.4, per_image=True)),
#             type='DiceLoss', use_sigmoid=False, loss_weight=0.4)),
# #             type='FocalLoss', loss_weight=0.4)),
#             type='CrossEntropyLoss', loss_weight=0.4)),


    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
