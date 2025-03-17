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
from dlavm.runtime import RuntimeBase

chin, chout = [4096, 4096]
token = ne.Var("token")
last_token = ne.Var("last_token")
f_head, w_head = [32, 2]

device = ohbm_accel.OHBM0314
init_addr = {"hbm": 0x0, "runtime": "hbm", "hbm_cache": "hbm"}
configs = {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0}
name, target = "test", targets.hpp


def ln_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [2*chin], dtype=de.fp16)
    output = dlavm.op.nn.layer_norm(input, weight)

    output = transform.infer_type(output, device)
    return output


output = ln_compare()
mod = backend.build(output, init_addr, name, False, target, configs)

rt = RuntimeBase(mod.lib)
print(mod.lib)
rt.main(name, token=19, last_token=0)

