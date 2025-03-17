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
import sys
import dlavm.utils
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000


def chatglm_block(input, last_token, pew, silu, index):
    prefix = "BLOCK%02d_" % index
    def const_hbm(name, data, shape, dtype=None):
        if dtype is None:
            return adr.const_hbm(prefix + name, prefix + data, shape)
        return adr.const_hbm(prefix + name, prefix + data, shape, dtype)
    lnw0 = const_hbm("lnweight0", "test", [2*4096], dtype=de.fp16)
    lnw1 = const_hbm("lnweight1", "test", [2*4096], dtype=de.fp16)
    qw   = const_hbm("qweight", "test", [4096, 4096])
    qb   = const_hbm("qbias", "test", [2*4096], dtype=de.fp16)
    kw   = const_hbm("kweight", "test", [256, 4096])
    kb   = const_hbm("kbias", "test", [2*256], dtype=de.fp16)
    vw   = const_hbm("vweight", "test", [256, 4096])
    vb   = const_hbm("vbias", "test", [2*256], dtype=de.fp16)
    ow   = const_hbm("oweight", "test", [4096, 4096])
    ob   = const_hbm("obias", "test", [2*4096], dtype=de.fp16)
    hw1  = const_hbm("hweight1", "test", [13696, 4096])
    hb1  = const_hbm("hbias1", "test", [2*13696], dtype=de.fp16)
    hw2  = const_hbm("hweight2", "test", [13696, 4096])
    hb2  = const_hbm("hbias2", "test", [2*13696], dtype=de.fp16)
    lw   = const_hbm("lweight", "test", [4096, 13696])
    lb   = const_hbm("lbias", "test", [2*4096], dtype=de.fp16)

    ln_out = dlavm.nn.rms_norm(input, lnw0)
    q_data = dlavm.nn.mvm_f16xi4(ln_out, qw, qb, out_heads=[32, 2])
    k_data = dlavm.nn.mvm_f16xi4(ln_out, kw, kb)
    v_data = dlavm.nn.mvm_f16xi4(ln_out, vw, vb)

    k_data = dlavm.reshape(k_data, new_shape=[2, -1, 128])
    v_data = dlavm.reshape(v_data, new_shape=[2, -1, 128])

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

token = ne.Var("token", 2048)
last_token = ne.Var("last_token", 2048)
token = 19
last_token = 0
device = ohbm_accel.OHBM0316

pew = adr.const_hbm("pos_emb_weight", "test", [256, 4096], dtype=de.fp16)
silu = adr.const_hbm("silu_weight", "test", [16*3], dtype=de.fp16)

outlnw = adr.const_hbm("out_lnweight", "test", [2*4096], dtype=de.fp16)
outw = adr.const_hbm("oweight", "test", [65024, 4096])
outb = adr.const_hbm("obias", "test", [2*65024], dtype=de.fp16)

input = adr.var_hbm("input", [1, token, 4096])

for i in range(1):
    input = chatglm_block(input, last_token, pew, silu, i)

out_ln = dlavm.nn.rms_norm(input, outlnw)
output = dlavm.nn.mvm_f16xi4(out_ln, outw, outb, argmax=True)
output = output[1]

output = transform.infer_type(output, device)
print(output)


if __name__ == "__main__":
    from dlavm.driver import config
    config.tb_sim_path = "/home/shenao/dlavm-llm-public/tbsim/workspace_2025_0301"

    init_addr = {"hbm": 0x0, "hbm_cache": "hbm", "runtime": "hbm_cache", "onchip": 0x0}
    # mod = backend.build_tb(output, init_addr, "test", targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
    mod = backend.build(output, init_addr, "test", False, targets.hpp, {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0})
    with open("chatglm_test_19_0316.h", "w") as f:
        print(mod.get_source(), file=f)
