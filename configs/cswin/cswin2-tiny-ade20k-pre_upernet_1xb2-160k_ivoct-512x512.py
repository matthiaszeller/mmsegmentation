_base_ = [
    './cswin2-tiny_upernet_1xb2-160k_ivoct-512x512.py'
]

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None)
)
