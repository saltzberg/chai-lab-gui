from pathlib import Path

# Base paths
BASE_DATA_DIR = Path("/data/Chai1")
DOWNLOADS_DIR = BASE_DATA_DIR / "downloads"
OUTPUT_DIR = BASE_DATA_DIR / "output"
TMP_DIR = Path("./tmp")

# Model paths and URLs
COMPONENT_URL = "https://chaiassets.com/chai1-inference-depencencies/models_v2/{comp_key}"
CONFORMERS_URL = "https://chaiassets.com/chai1-inference-depencencies/conformers_v1.apkl"

# Default parameters
DEFAULT_NUM_TRUNK_RECYCLES = 3
DEFAULT_NUM_DIFFN_TIMESTEPS = 200
DEFAULT_SEED = 42 