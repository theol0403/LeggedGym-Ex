import os
import sys

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
print(f"LEGGED_GYM_ROOT_DIR: {LEGGED_GYM_ROOT_DIR}")
LEGGED_GYM_ENVS_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, 'legged_gym', 'envs')
PROJECT_CACHE_DIR = os.path.join(LEGGED_GYM_ROOT_DIR, ".cache")
GENESIS_CACHE_DIR = os.path.join(PROJECT_CACHE_DIR, "genesis")

if sys.version_info[1] >= 10: # >=3.10 for genesis and isaacsim
    simulator_type = os.getenv("SIMULATOR")
    if simulator_type == "genesis":
        os.environ.setdefault("XDG_CACHE_HOME", PROJECT_CACHE_DIR)
        os.environ.setdefault("GS_CACHE_FILE_PATH", GENESIS_CACHE_DIR)
        SIMULATOR = "genesis"
    elif simulator_type == "isaaclab":
        SIMULATOR = "isaaclab"
    else:
        raise ValueError("Unsupported SIMULATOR type. Please set the SIMULATOR environment variable to 'genesis' or 'isaaclab'.")
elif sys.version_info[1] <= 8 and sys.version_info[1] >= 6: # >=3.6 and <3.9 for isaacgym
    SIMULATOR = "isaacgym"

if SIMULATOR == "genesis":
    try: 
        import genesis as gs
    except ImportError as e:
        print("Failed to import Genesis. Please ensure that the Genesis is properly installed and configured.")
        raise e
elif SIMULATOR == "isaacgym":
    try:
        import isaacgym
    except ImportError as e:
        print("Failed to import Isaac Gym. Please ensure that the Isaac Gym is properly installed and configured.")
        raise e
elif SIMULATOR == "isaaclab":
    try:
        import isaaclab
    except ImportError as e:
        print("Failed to import Isaac Lab. Please ensure that the Isaac Lab is properly installed and configured.")
        raise e
