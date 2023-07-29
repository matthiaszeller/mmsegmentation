_base_ = [
    './cswin-tiny-ade20k-pre_1xb16-20k_ivoct-aug-512x512.py'
]

# need to download manually https://github.com/microsoft/CSWin-Transformer/releases/download/v0.2.0/upernet_cswin_tiny.pth
# and use tools/converters/cswin2mmseg.py
# then load with --cfg-options model.init_cfg.checkpoint=...
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None),
    decode_head=dict(
        channels=256
    )
)
