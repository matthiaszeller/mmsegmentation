import hashlib
import os
import re
from pathlib import Path
from time import time
from urllib.parse import urlparse

import requests
import os.path as osp

import torch.cuda
from tqdm import tqdm

# This should work out of the box with correct package versions and repo commit
from mmseg.apis import inference_segmentor, init_segmentor


class ModelFactory:
    @staticmethod
    def compute_file_hash(filename, hash_type):
        hash_obj = hashlib.new(hash_type)
        with open(filename, 'rb') as f:
            while True:
                data = f.read(8192)
                if not data:
                    break
                hash_obj.update(data)
        return hash_obj.hexdigest()

    @classmethod
    def check_checkpoint_hash(cls, ckpt_file) -> bool:
        full_hash = cls.compute_file_hash(ckpt_file, 'sha256')
        cur_hash = full_hash[:8]
        true_hash_code = Path(ckpt_file).stem.split('-')[-1]
        return true_hash_code == cur_hash

    @staticmethod
    def download_file(url, filename):
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(filename, 'wb') as f:
            file_size = int(response.headers.get('Content-Length', 0))
            progress = tqdm(response.iter_content(1024), f'Downloading {filename}', total=file_size, unit='B',
                            unit_scale=True, unit_divisor=1024)
            for data in progress:
                f.write(data)
                progress.update(len(data))

    @classmethod
    def download_checkpoint(cls, url: str, directory):
        parsed = urlparse(url)
        ckpt_name = Path(parsed.path).name
        ckpt_file = Path(directory) / ckpt_name

        if ckpt_file.exists():
            print('file exists, verifying checksum')
            ok = cls.check_checkpoint_hash(ckpt_file)
            if not ok:
                print('invalid hash, downloading')
            else:
                print('valid hash')
                return

        cls.download_file(url, ckpt_file)
        ok = cls.check_checkpoint_hash(ckpt_file)
        assert ok, 'failed download'

    @staticmethod
    def list_checkpoints():
        def parse_url(url: str) -> dict[str]:
            urlpath = urlparse(url).path
            *_, model_name, config_name, ckpt_name = Path(urlpath).parts
            return {'model_name': model_name, 'config_name': config_name, 'ckpt_name': ckpt_name, 'url': url}

        readmes = Path(__file__).parent.glob('configs/*/README.md')
        regexp = re.compile(r'\[model\]\((.+?)\)')
        urls = []
        for readme in readmes:
            with open(readme) as f:
                urls.extend(regexp.findall(f.read()))

        return list(map(parse_url, urls))


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

config_file = f'configs/{model_name}/{config_name}.py'
checkpoint_file = f'../{checkpoint_name}'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device', device)
# build the model from a config file and a checkpoint file
model = init_segmentor(config_file, checkpoint_file, device=device)

# test a single image and show the results
img = 'demo/demo_oct.jpg'  # or img = mmcv.imread(img), which will only load it once
dt = time()
result = inference_segmentor(model, img)
# visualize the results in a new window
model.show_result(img, result, out_file='out.jpg')
dt = time() - dt
print(f'written in file out.jpg, took {dt:.4} s')

###



