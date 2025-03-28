import os
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
from time import strftime, localtime
import dlavm.utils
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000


def qwen2_block(input, last_token, pew, silu, index):
    prefix = "BLOCK%02d_" % index
    def const_hbm(name, data, shape, dtype=None):
        if dtype is None:
            return adr.const_hbm(prefix + name, prefix + data, shape)
        return adr.const_hbm(prefix + name, prefix + data, shape, dtype)
    lnw0 = const_hbm("lnweight0", "test", [2*3584], dtype=de.fp16)
    lnw1 = const_hbm("lnweight1", "test", [2*3584], dtype=de.fp16)
    qw   = const_hbm("qweight", "test", [3584, 3584])
    qb   = const_hbm("qbias", "test", [2*3584], dtype=de.fp16)
    kw   = const_hbm("kweight", "test", [128*4, 3584])
    kb   = const_hbm("kbias", "test", [2*128*4], dtype=de.fp16)
    vw   = const_hbm("vweight", "test", [128*4, 3584])
    vb   = const_hbm("vbias", "test", [2*128*4], dtype=de.fp16)
    ow   = const_hbm("oweight", "test", [3584, 3584])
    ob   = const_hbm("obias", "test", [2*3584], dtype=de.fp16)
    hw1  = const_hbm("hweight1", "test", [18944, 3584])
    hb1  = const_hbm("hbias1", "test", [2*18944], dtype=de.fp16)
    hw2  = const_hbm("hweight2", "test", [18944, 3584])
    hb2  = const_hbm("hbias2", "test", [2*18944], dtype=de.fp16)
    lw   = const_hbm("lweight", "test", [3584, 18944])
    lb   = const_hbm("lbias", "test", [2*3584], dtype=de.fp16)

    ln_out = dlavm.nn.rms_norm(input, lnw0)
    q_data = dlavm.nn.mvm_f16xi4(ln_out, qw, qb, out_heads=[28, 4])
    k_data = dlavm.nn.mvm_f16xi4(ln_out, kw, kb)
    v_data = dlavm.nn.mvm_f16xi4(ln_out, vw, vb)

    k_data = dlavm.reshape(k_data, new_shape=[4, -1, 128])
    v_data = dlavm.reshape(v_data, new_shape=[4, -1, 128])

    q_data = dlavm.nn.rope_glm(q_data, pew, last_token=last_token)
    k_data = dlavm.nn.rope_glm(k_data, pew, last_token=last_token)

    k_data = dlavm.nn.kcache2hbm(k_data, cache_len=last_token)
    v_data = dlavm.nn.vcache2hbm(v_data, cache_len=last_token)

    qk_data = dlavm.nn.mvm_f16xf16(q_data, k_data, w_trp=True)
    qk_data = dlavm.nn.softmax(qk_data)
    o_data  = dlavm.nn.mvm_f16xf16(qk_data, v_data)

    atten_data = dlavm.nn.mvm_f16xi4(o_data, ow, ob)
    atten_data = dlavm.nn.add(atten_data, input)

    ln_out    = dlavm.nn.rms_norm(atten_data, lnw1)
    h1_data   = dlavm.nn.mvm_f16xi4(ln_out, hw1, hb1)
    h2_data   = dlavm.nn.mvm_f16xi4(ln_out, hw2, hb2)
    silu_data = dlavm.nn.activate(h1_data, silu)
    h_data    = dlavm.nn.mul(silu_data, h2_data)
    l_data    = dlavm.nn.mvm_f16xi4(h_data, lw, lb)
    block_out = dlavm.nn.add(l_data, atten_data)
    return block_out

# token = 19
token = ne.Var("token", 2048)
last_token = ne.Var("last_token", 2048)

pew = adr.const_hbm("pos_emb_weight", "test", [2048, 3584], dtype=de.fp16)
silu = adr.const_hbm("silu_weight", "test", [16*3], dtype=de.fp16)

outlnw = adr.const_hbm("out_lnweight", "test", [2*3584], dtype=de.fp16)
outw = adr.const_hbm("oweight", "test", [152064, 3584])
outb = adr.const_hbm("obias", "test", [2*152064], dtype=de.fp16)

input = adr.var_hbm("input", [1, token, 3584])

for i in range(1):
    input = qwen2_block(input, last_token, pew, silu, i)

input = dlavm.gather(input)
out_ln = dlavm.nn.rms_norm(input, outlnw)
output = dlavm.nn.mvm_f16xi4(out_ln, outw, outb, argmax=True)
output = output[1]

device = ohbm_accel.OHBM0326V80
output = transform.infer_type(output, device)
print(output)


if __name__ == "__main__":
    from dlavm.driver import config
    config.tb_sim_path = device.tb_sim_path
    config.sim_tool = "modelsim"

    name = f"qwen2_v80_debug_ohbm"
    name += "_" + strftime('%m%d_%H%M', localtime())

    init_addr = {"hbm": 0x0, "hbm_cache": "hbm", "runtime": "hbm_cache", "onchip": 0x0}
    # mod = backend.build_tb(output, init_addr, "test", targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
    mod = backend.build(output, init_addr, "test", False, targets.v80, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0, "align": 0x1000, "addr_dtype": "uint64_t"})

    src = os.path.join("output", name+".h")
    ptx = os.path.join("output", name+".prototxt")
    log = os.path.join("output", name+".log")
    with open(src, "w") as f:
        print(mod.get_source(), file=f)
    with open(ptx, "w") as f:
        print(mod.get_prototxt(), file=f)