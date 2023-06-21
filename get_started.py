

from time import time

import torch.cuda

# This should work out of the box with correct package versions and repo commit
from mmseg.apis import init_model, inference_model, show_result_pyplot


def display_nested_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        print('\t' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            display_nested_dict(value, indent + 1)
        else:
            print(value)

# Very basic model
# model_name, config_name, checkpoint_name = (
#     'pspnet',
#     'pspnet_r50-d8_4xb4-80k_ade20k-512x512',
#     'pspnet_r50-d8_512x512_80k_ade20k_20200615_014128-15a8b914.pth'
# )

# Swin
model_name, config_name, checkpoint_name = (
    'swin',
    'swin-tiny-patch4-window7-in1k-pre_upernet_8xb2-160k_ade20k-512x512',
    'upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
)

# # CSWin migrated to newer version
# model_name, config_name, checkpoint_name = (
#     'cswin',
#     'cswin-tiny-patch4-split1277-in1k-pre_upernet_160k_ade20k-512x512',
#     'converted.pth'
# )

# Custom CSWin with its own checkpoint
# model_name, config_name, checkpoint_name = (
#     'cswin',
#     'upernet_cswin_tiny_res_rpe_5_res1',
#     'iter_153000.pth'
# )

# Custom CSWin with original CSWin checkpoint
# model_name, config_name, checkpoint_name = (
#     'cswin',
#     'upernet_cswin_tiny_res_rpe_5_res1',
#     '../cswin-tiny-patch4-split1277-in1k-pre_upernet_160k_ade20k-512x512/cswin-tiny-patch4-split1277-in1k-pre_upernet_160k_ade20k-512x512-a4d67e45.pth'
# )

config_file = f'configs/{model_name}/{config_name}.py'
checkpoint_file = f'work_dirs/{config_name}/{checkpoint_name}'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device', device)
# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device=device)
print('Model config:')
display_nested_dict(model.cfg)

# test a single image and show the results
img = 'demo/demo_oct.jpg'  # or img = mmcv.imread(img), which will only load it once
dt = time()
result = inference_model(model, img)
# visualize the results
show_result_pyplot(model, img, result, show=False, out_file='out.jpg', opacity=0.5)

dt = time() - dt
print(f'written in file out.jpg, took {dt:.4} s')

###



