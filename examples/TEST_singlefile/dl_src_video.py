"""
dl_src_video.py - Download a single file from dropbox.

link: https://www.dropbox.com/s/61ppfw0dwtw8dct/President%20John%20F.%20Kennedy%27s%20Peace%20Speech.mp4?dl=1

Video originally downloaded from C-SPAN and is public domain.
"""
import requests
import os
from os.path import join


single_test = {
    "President John F. Kennedy's Peace Speech.mp4": "https://www.dropbox.com/s/61ppfw0dwtw8dct/President%20John%20F.%20Kennedy%27s%20Peace%20Speech.mp4?dl=1"
}


def download_single_file(link, filename):
    """Download a single file from a remote server."""
    local_name = join(os.getcwd(), "examples", "TEST_singlefile", filename)
    if os.path.exists(local_name):
        print(f"File {filename} already exists. Skipping download.")
        return
    print("Downloading file...")
    with open(local_name, "wb") as f:
        f.write(requests.get(link).content)
    print("Download complete.")


if __name__ == "__main__":
    for filename, link in single_test.items():
        download_single_file(link, filename)
