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

# token = 19
token = ne.Var("token", 2048)
input = adr.var_hbm("input", [1, token, 4096])
weight = adr.const_hbm("weight1", "test", [2*4096], dtype=de.fp16)
# bn = adr.const_hbm("bn", "test", [2*4096], dtype=adr.DataEnum.fp16)

output = dlavm.op.nn.rms_norm(input, weight)

output = transform.infer_type(output, ohbm_accel.OHBM)
print(output)

from dlavm.driver import config
config.tb_sim_path = "/home/shenao/dlavm-llm-public/tbsim/workspace_2025_0303"

init_addr = {"hbm": 0x0, "runtime": "hbm", "hbm_cache": "hbm"}
# mod = backend.build_tb(output, init_addr, "test", targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
# print(mod.get_source())
mod = backend.build(output, init_addr, "test", False, targets.hpp, {"wt2hbm":False, "hbm_base": 0x100, "ddr_base": 0x0})
print(mod.get_source())

