_base_ = [
    './cswin-tiny_1xb16-20k_ivoct-512x512.py', 
]

crop_size = (512, 512)
data_preprocessor = dict(size=crop_size)

# need to download manually https://github.com/microsoft/CSWin-Transformer/releases/download/v0.2.0/upernet_cswin_tiny.pth
# and use tools/converters/cswin2mmseg.py
# then load with --cfg-options model.init_cfg.checkpoint=...
model = dict(
    data_preprocessor=data_preprocessor,
    init_cfg=dict(type='Pretrained', checkpoint=None),
)
