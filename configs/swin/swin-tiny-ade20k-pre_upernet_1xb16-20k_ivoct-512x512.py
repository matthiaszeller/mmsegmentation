_base_ = [
    './swin-tiny_1xb16-20k_ivoct-512x512.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/swin/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K/upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'  # noqa
model = dict(
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
)
