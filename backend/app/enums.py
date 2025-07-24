from enum import Enum

class Backend(str, Enum):
    local = "local"
    bash = "bash"
    slurm = "slurm" 