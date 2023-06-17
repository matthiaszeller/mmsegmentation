from pathlib import Path

from mmseg.apis import inference_segmentor, init_segmentor
from tqdm import tqdm

import argparse


def main(work_dir: Path, output_dir: Path, img_dir: Path):
    pass


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('work_dir', type=Path)
    p.add_argument('output_dir', type=Path)
    p.add_argument('img_dir', type=Path)
    args = p.parse_args()

    main(args)
