_base_ = [
    './cswin2-tiny_upernet_20k_ivoct-512x512.py'
]

model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None)
)
