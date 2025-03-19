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
from dlavm.utils.tools import RegsCheckSame


chin, chout = [4096, 4096]
token = 19
last_token = 0
f_head, w_head = [32, 2]
device = ohbm_accel.OHBM0316

from dlavm.driver import config
config.tb_sim_path = device.tb_sim_path

init_addr = {"hbm": 0x0, "runtime": "hbm", "hbm_cache": "hbm"}
configs = {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0, "min_loop": -1}
name, target = "test", targets.hpp


def run_check(fn):
    name = fn.__name__
    start = f" Check Start: \033[36;34m{name}\033[36;0m "
    print(f"{start:=^135}")
    graph, mod1, mod2, ignores = fn()
    print(graph)
    regs1 = mod1.reg_serialization()
    regs2 = mod2.reg_serialization()
    if RegsCheckSame(regs1, regs2, ignores):
        success = f"\033[36;32m Check Success!\033[36;0m "
        finish = f" Check Finish: \033[36;32m{name}\033[36;0m "
        print(f"{success:-^135}")
    else:
        fail = f"\033[36;31m Check Fail! \033[36;0m"
        finish = f" Check Finish: \033[36;31m{name}\033[36;0m "
        print(f"{fail:*^135}")
    print(f"{finish:=^135}\n")


def run_expr_check(fn):
    name = fn.__name__
    start = f" Check Start: \033[36;34m{name}\033[36;0m "
    print(f"{start:=^135}")
    expr, ignores = fn()
    output = transform.infer_type(expr, device)
    print(output)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    regs1 = mod1.reg_serialization()
    regs2 = mod2.reg_serialization()
    if RegsCheckSame(regs1, regs2, ignores):
        success = f"\033[36;32m Check Success!\033[36;0m "
        finish = f" Check Finish: \033[36;32m{name}\033[36;0m "
        print(f"{success:-^135}")
    else:
        fail = f"\033[36;31m Check Fail! \033[36;0m"
        finish = f" Check Finish: \033[36;31m{name}\033[36;0m "
        print(f"{fail:*^135}")
    print(f"{finish:=^135}\n")


# @run_expr_check
def mvm_out_heads_atten_compare():
    qw   = adr.const_hbm("qweight", "test", [4096, 4096])
    qb   = adr.const_hbm("qbias", "test", [2*4096], dtype=de.fp16)

    ln_out = adr.var_hbm("input", [1, token, 4096])
    output = dlavm.nn.mvm_f16xi4(ln_out, qw, qb, out_heads=[32, 2])
    return output, []


# @run_expr_check
def glm_atten_compare():
    pew = adr.const_hbm("pos_emb_weight", "test", [256, 4096], dtype=de.fp16)
    qw   = adr.const_hbm("qweight", "test", [4096, 4096])
    qb   = adr.const_hbm("qbias", "test", [2*4096], dtype=de.fp16)
    kw   = adr.const_hbm("kweight", "test", [256, 4096])
    kb   = adr.const_hbm("kbias", "test", [2*256], dtype=de.fp16)
    vw   = adr.const_hbm("vweight", "test", [256, 4096])
    vb   = adr.const_hbm("vbias", "test", [2*256], dtype=de.fp16)
    ow   = adr.const_hbm("oweight", "test", [4096, 4096])
    ob   = adr.const_hbm("obias", "test", [2*4096], dtype=de.fp16)

    ln_out = adr.var_hbm("input", [1, token, 4096])
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
    output = dlavm.nn.mvm_f16xi4(o_data, ow, ob)

    return output, []


@run_check
def conv2d_compare():
    hin, win, chin, chout = 224, 224, 512, 512
    k = 3
    p = 0
    s = 2
    input = adr.var_hbm("input", [hin, win, chin])
    weight = adr.const_hbm("weight1", "test", [k, k, chin, chout])
    bn = adr.const_hbm("weight2", "test", [2*chout], dtype=de.fp16)
    output = dlavm.op.nn.conv2d(input, weight, bn, padding=[p, p], strides=[s, s])

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [10, 11, 13, 25]


# @run_check
def kcache_compare():
    input_k = adr.var_hbm("input_k", [w_head, token, 128])
    output = dlavm.op.nn.kcache2hbm(input_k, cache_len=last_token)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [131, 134]


# @run_check
def vcache_compare():
    input_v = adr.var_hbm("input_v", [w_head, token, 128])
    output = dlavm.op.nn.vcache2hbm(input_v, cache_len=last_token)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [131, 134]


# @run_check
def trp_mvm_compare():
    input_q = adr.var_hbm("input_q", [f_head, token, 128])
    input_k = adr.var_hbm("input_k", [w_head, token+last_token, 128])
    output = dlavm.op.nn.mvm_f16xf16(input_q, input_k, w_trp=True)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [10, 11, 13]


# @run_check
def f2w_mvm_compare():
    input_a = adr.var_hbm("input_a", [f_head, token, token+last_token])
    input_v = adr.var_hbm("input_v", [w_head, token+last_token, 128])
    output = dlavm.op.nn.mvm_f16xf16(input_a, input_v)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [10, 11, 13]


# @run_check
def ln_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [2*chin], dtype=de.fp16)
    output = dlavm.op.nn.layer_norm(input, weight)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [130, 131, 134]


# @run_check
def rms_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [2*chin], dtype=de.fp16)
    output = dlavm.op.nn.rms_norm(input, weight)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [130, 131, 134]


# @run_check
def mvm_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight", "test", [chout, chin])
    output = dlavm.op.nn.mvm_f16xi4(input, weight)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [10, 11, 13, 25]


# @run_check
def mvm_bn_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight", "test", [chout, chin])
    bn = adr.const_hbm("bn", "test", [2*chout], dtype=de.fp16)
    output = dlavm.op.nn.mvm_f16xi4(input, weight, bn)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [10, 11, 13, 25]


# @run_check
def mvm_bn_argmax_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight", "test", [chout, chin])
    bn = adr.const_hbm("bn", "test", [2*chout], dtype=de.fp16)
    output = dlavm.op.nn.mvm_f16xi4(input, weight, bn, argmax=True)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [10, 11, 13, 25]


# @run_check
def softmax_compare():
    input = adr.var_hbm("input", [f_head, token, token+last_token])
    output = dlavm.op.nn.softmax(input)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [131, 134]


# @run_check
def elementwise_compare():
    input1 = adr.var_hbm("input1", [1, token, chin])
    input2 = adr.var_hbm("input2", [1, token, chin])
    output = dlavm.op.nn.add(input1, input2)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [131, 134, 140]


# @run_check
def activate_compare():
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [chin], dtype=de.fp16)
    output = dlavm.op.nn.activate(input, weight)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [130, 131, 134]


# @run_check
def emb_glm_compare():
    input = adr.var_hbm("input", [f_head, token, 128])
    weight = adr.const_hbm("weight1", "test", [100, 128], dtype=de.fp16)
    output = dlavm.op.nn.rope_glm(input, weight, last_token=last_token)

    output = transform.infer_type(output, device)
    mod1 = backend.build_tb(output, init_addr, name, target, configs)
    mod2 = backend.build(output, init_addr, name, False, target, configs)
    return output, mod1, mod2, [130, 131, 134]
