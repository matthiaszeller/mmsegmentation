_base_ = [
    './cswin-tiny_1xb16-20k_ivoct-512x512.py'
]

# need to download manually https://github.com/microsoft/CSWin-Transformer/releases/download/v0.1.0/cswin_tiny_224.pth
# then use tools/converters/cswin2mmseg.py
checkpoint_file = './work_dirs/cswin-tiny-in1k-pre_1xb16-20k_ivoct-512x512/cswin_tiny_224_converted.pth'
model = dict(
    backbone=dict(
        init_cfg=dict(_delete_=True, type='Pretrained', checkpoint=checkpoint_file),
    ),
)
