_base_ = [
    './cswin-tiny_1xb16-20k_ivoct-512x512.py'
]

data_preprocessor = dict(
    mean=[23.45 ,  5.507,  1.78 ],
    std=[30.345, 16.228,  2.11],
)

# need to download manually https://github.com/microsoft/CSWin-Transformer/releases/download/v0.2.0/upernet_cswin_tiny.pth
# and use tools/converters/cswin2mmseg.py
# then load with --cfg-options model.init_cfg.checkpoint=...
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=None),
    data_preprocessor=data_preprocessor
)
