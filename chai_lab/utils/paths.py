# Copyright (c) 2024 Chai Discovery, Inc.
# Licensed under the Apache License, Version 2.0.
# See the LICENSE file for details.

import dataclasses
import os
import random
from pathlib import Path
from typing import Final

import requests
from filelock import FileLock
from chai_lab.config import DOWNLOADS_DIR, COMPONENT_URL, CONFORMERS_URL

# use this path object to specify location
# of anything within repository
repo_root: Final[Path] = Path(__file__).parents[2].absolute()

# weights and helper data is downloaded to DOWNLOADS_DIR
downloads_path = DOWNLOADS_DIR

# minimal sanity check in case we start moving things around
assert repo_root.exists()

def download_if_not_exists(http_url: str, path: Path):
    if path.exists():
        return

    with FileLock(path.with_suffix(".download_lock")):
        if path.exists():
            return  # if-lock-if sandwich to download only once
        print(f"downloading {http_url}")
        tmp_path = path.with_suffix(f".download_tmp_{random.randint(10 ** 5, 10**6)}")
        with requests.get(http_url, stream=True) as response:
            response.raise_for_status()
            path.parent.mkdir(exist_ok=True, parents=True)
            with tmp_path.open("wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        file.write(chunk)
    tmp_path.rename(path)
    assert path.exists()

@dataclasses.dataclass
class Downloadable:
    url: str
    path: Path

    def get_path(self) -> Path:
        download_if_not_exists(self.url, path=self.path)
        return self.path

cached_conformers = Downloadable(
    url=CONFORMERS_URL,
    path=downloads_path.joinpath("conformers_v1.apkl"),
)

def chai1_component(comp_key: str) -> Path:
    """
    Downloads exported model, stores in locally in the repo/downloads
    comp_key: e.g. 'trunk.pt'
    """
    assert comp_key.endswith(".pt")
    url = COMPONENT_URL.format(comp_key=comp_key)
    result = downloads_path.joinpath("models_v2", comp_key)
    download_if_not_exists(url, result)
    return result
