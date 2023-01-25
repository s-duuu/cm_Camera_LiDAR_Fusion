from .voxel_set_abstraction import VoxelSetAbstraction
import os
import sys
lib_path = os.path.abspath(os.path.join(__file__, '..'))
sys.path.insert(0, lib_path)

__all__ = {
    'VoxelSetAbstraction': VoxelSetAbstraction
}
