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

# token = ne.Var("token", 2048)
token = 19
chin, chout = 1024, 1024
input = adr.var_hbm("input", [1, token, chin])
weight = adr.const_hbm("weight1", "test", [chout, chin])
# bn = adr.const_hbm("bn", "test", [2*4096], dtype=adr.DataEnum.fp16)

output = dlavm.op.nn.mvm_f16xi4(input, weight)

output = transform.infer_type(output, ohbm_accel.OHBM)
print(output)

from dlavm.driver import config
config.tb_sim_path = "/home/shenao/dlavm-llm-public/tbsim/workspace_2025_0303"

init_addr = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "insts": "runtime", "hbm": 0x0, "hbm_cache": "hbm", "hbm_rt": "hbm_cache", "onchip": 0x0}
mod = backend.build_tb(output, init_addr, "test", targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
# mod = backend.build(output, init_addr, "test", False, targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
with open(f"mvm_{token}_{chin}_{chout}.h", "w") as f:
    print(mod.get_source(), file=f)

