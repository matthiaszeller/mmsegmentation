from pathlib import Path
from time import time

import torch.cuda

# This should work out of the box with correct package versions and repo commit
from mmseg.apis import inference_segmentor, init_segmentor


def display_nested_dict(dictionary, indent=0):
    for key, value in dictionary.items():
        print('\t' * indent + str(key) + ':', end=' ')
        if isinstance(value, dict):
            print()
            display_nested_dict(value, indent + 1)
        else:
            print(value)


# model_name, config_name, checkpoint_name = (
#     'pspnet',
#     'pspnet_r50-d8_512x1024_40k_cityscapes',
#     'pspnet_r50-d8_512x1024_40k_cityscapes_20200605_003338-2966598c.pth'
# )
# model_name, config_name, checkpoint_name = (
#     'swin',
#     'upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K',
#     'upernet_swin_tiny_patch4_window7_512x512_160k_ade20k_pretrain_224x224_1K_20210531_112542-e380ad3e.pth'
# )
model_name, config_name, checkpoint_name = (
    'cswin',
    'upernet_cswin_tiny_res_rpe_5_res1',
    'iter_153000.pth'
)

config_file = str(Path('configs').joinpath(model_name, config_name).with_suffix('.py'))
checkpoint_file = str(Path('work_dirs').joinpath(config_name, checkpoint_name))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device', device)
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device=device)
print('Model config:')
display_nested_dict(model.cfg)

# test a single image and show the results
img = 'demo/demo_oct.jpg'  # or img = mmcv.imread(img), which will only load it once
dt = time()
result = inference_segmentor(model, img)
# visualize the results in a new window
model.show_result(img, result, out_file='out.jpg')
dt = time() - dt
print(f'written in file out.jpg, took {dt:.4} s')

###



