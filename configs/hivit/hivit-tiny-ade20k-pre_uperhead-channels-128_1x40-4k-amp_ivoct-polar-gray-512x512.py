_base_ = [
    './hivit-tiny_uperhead-channels-128_1x40-4k-amp_ivoct-polar-gray-512x512.py'
]


model = dict(
    # use --cfg-options model.init_cfg.checkpoint at runtime
    init_cfg=dict(type='Pretrained', checkpoint=None)
)
