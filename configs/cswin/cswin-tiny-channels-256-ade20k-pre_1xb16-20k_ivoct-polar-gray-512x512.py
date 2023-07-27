_base_ = [
    './cswin-tiny-ade20k-pre_1xb16-20k_ivoct-polar-gray-512x512.py'
]

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None),
        decode_head=dict(
        channels=256
    )
)

