from nyxus import Nyxus 
import nyxus
import time

path = '/home/ec2-user/Dev/nyxus_backup_6_13/nyxus/build/tests/dsb2018/train'

intens = path + '/images'
seg = path + '/masks'

#nyx = Nyxus(["GABOR"])
print(nyxus.gpu_is_available())
print(nyxus.get_gpu_properties())

# cpu gabor
start = time.time()
cpu_nyx = Nyxus(["GABOR"], n_feature_calc_threads=8)
cpu_nyx.using_gpu(False)
cpu_features = cpu_nyx.featurize(intens, seg)
end = time.time()
print(cpu_features)
print("cpu time: " + str(end-start))

# gpu gabor
start = time.time()
gpu_nyx = Nyxus(["GABOR"], n_feature_calc_threads=1)
gpu_nyx.using_gpu(True)
gpu_features = gpu_nyx.featurize(intens, seg)
end = time.time()
print(gpu_features)
print("gpu time: " + str(end-start))