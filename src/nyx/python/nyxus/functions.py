from .backend import gpuIsAvailable, getGpuProperties

def gpu_is_available():
    return gpuIsAvailable()

def get_gpu_properties():
    return getGpuProperties()