"""
dl_src_video.py - Download source videos from dropbox.

All videos are under creative commons license and are cited in CITATIONS file in this working directory.

link to folder containing all videos : https://www.dropbox.com/sh/0u3easov5bygo5l/AACXWZ_gXhAsSTHg64FYsjf5a?dl=0

In case of dropbox server issues, please refer to original URLs in CITATIONS file.
"""
import requests
import os
from os.path import join
from tqdm.auto import tqdm

multi_test = {
    "MIT_MatricesSGD_000.mp4": "https://www.dropbox.com/s/9b0j41zf3jaswv2/MIT_MatricesSGD_000.mp4?dl=1",
    "MIT_MatricesSGD_001.mp4": "https://www.dropbox.com/s/691qsessxsbzbvz/MIT_MatricesSGD_001.mp4?dl=1",
    "MIT_MatricesSGD_002.mp4": "https://www.dropbox.com/s/tqzn5ocnx0rghw7/MIT_MatricesSGD_002.mp4?dl=1",
    "MIT_Signals_000.mp4": "https://www.dropbox.com/s/bgyk0azwp67rbvn/MIT_Signals_000.mp4?dl=1",
    "MIT_Signals_001.mp4": "https://www.dropbox.com/s/uufiecjos44vh4d/MIT_Signals_001.mp4?dl=1",
    "MIT_Signals_002.mp4": "https://www.dropbox.com/s/zuyiqbqw2fn3tb8/MIT_Signals_002.mp4?dl=1",
    "MIT_VibrationsAndWaves000.mp4": "https://www.dropbox.com/s/laeo90gjdlqffn2/MIT_VibrationsAndWaves000.mp4?dl=1",
    "MIT_VibrationsAndWaves001.mp4": "https://www.dropbox.com/s/8s9vwohtlyseoq9/MIT_VibrationsAndWaves001.mp4?dl=1",
    "MIT_VibrationsAndWaves002.mp4": "https://www.dropbox.com/s/7rpg5nxcvzd0m9x/MIT_VibrationsAndWaves002.mp4?dl=1",
    "MIT_VibrationsAndWaves003.mp4": "https://www.dropbox.com/s/whwe8qydcnmkgf1/MIT_VibrationsAndWaves003.mp4?dl=1",
}  # contains all videos for the example


def download_single_file(link, filename):
    """Download a single file from a remote server."""
    local_name = join(os.getcwd(), "examples", "TEST_folder_edition", filename)
    if os.path.exists(local_name):
        print(f"\nFile {filename} already exists. Skipping download.")
        return
    print("\nDownloading file...")
    with open(local_name, "wb") as f:
        f.write(requests.get(link).content)
    print(f"\nDownload of {filename} complete.")


if __name__ == "__main__":
    pbar = tqdm(total=multi_test.__len__(), desc="Downloading example videos")
    for filename, link in multi_test.items():
        download_single_file(link, filename)
        pbar.update(1)
    pbar.close()
