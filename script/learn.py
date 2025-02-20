import sys
sys.path.append("..")

from dlavm import ne
from dlavm import adr

token = ne.Var("token", max_data=2048) # 动态维度定义，2048指此值最大为2048，以方便空间地址计算
a = adr.hbm.var_ddr(name="a", shape=(1, token, 4096)) # 定义一个ddr的变量，也即input，名字为a，形状为(1, token, 4096)
b = adr.hbm.const_hbm(name="b", data="hbm_weight.bin", shape=(4096, 256)) # 定义一个hbm类型的权重，名字叫b，shape方向为(CHin, CHout)
c = adr.hbm.mvm(a, b) # 定义一个矩阵乘法运算，结果为c

print(c)

from dlavm import device
from dlavm import transform

last_token = ne.Var("last_token", max_data=2048) # 定义last_token，统一加载到所有计算图的节点中
c = transform.infer_type(c, device.hbm_accel.HBM0923, attrs={"last_token": last_token})

print(c)

from dlavm import backend
from dlavm.target import targets

target = targets.hpp
init_addr = {"global": 0x0, "weight": "global", "runtime": "weight", "insts": "runtime", "hbm": 0x0, "hbm_cache": "hbm"}
build_config = {"wt2hbm": True, "debug": True, "ddr_base": 0x20000_0000, "hbm_base": 0x0, "align": 0x4000}
mod = backend.build(c, init_addr, "model_mvm", True, target, build_config)

source = mod.get_source()
insts = mod.get_insts_bin()

with open("src.inc.h", "w") as f:
    f.write(source)

with open("inst.bin", "wb") as f:
    f.write(insts)