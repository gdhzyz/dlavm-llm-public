import sys

sys.path.append("../..")
sys.setrecursionlimit(3000)

import dlavm
from dlavm import ne
from dlavm import adr
from dlavm import transform
from dlavm import backend
from dlavm.target import targets
from dlavm.device import ohbm_accel, hbm_accel
from dlavm.adr import DataEnum as de

# token = ne.Var("token", 2048)
token = 19
chin, chout = 16, 1024
input = adr.var_hbm("input", [256, 256, chin])
weight = adr.const_hbm("weight1", "test", [3, 3, chin, chout])
bn = adr.const_hbm("weight2", "test", [2*chout], dtype=de.fp16)

output = dlavm.op.nn.conv2d(input, weight, bn)

device = ohbm_accel.OHBM0316

from dlavm.driver import config
config.tb_sim_path = device.tb_sim_path

output = transform.infer_type(output, device)
print(output)


init_addr = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "insts": "runtime", "hbm": 0x0, "hbm_cache": "hbm", "hbm_rt": "hbm_cache", "onchip": 0x0}
mod = backend.build_tb(output, init_addr, "test", targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
# mod = backend.build(output, init_addr, "test", False, targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
with open(f"output/conv2d_tp.h", "w") as f:
    print(mod.get_source(), file=f)

