_base_ = [
    './vit_vit-b16-mae-ivoct-pre_mln_upernet_1xb16-20k_ivoct-polar-gray-512x512.py'
]


model = dict(
    decode_head=dict(
        channels=256,
    ),
)
