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
from dlavm.utils.tools import RegsCheckSameList


int_token = 213
int_last_token = 37
chin, chout = [4096, 4096]
f_head, w_head = [32, 2]
device = ohbm_accel.OHBM0326

from dlavm.driver import config
config.tb_sim_path = device.tb_sim_path
config.sim_tool = "modelsim"

str_token = "token"
str_last_token = "last_token"
token = ne.Var(str_token)
last_token = ne.Var(str_last_token)
init_addr = {"hbm": 0x0, "runtime": "hbm", "hbm_cache": "hbm"}
configs = {"wt2hbm":False, "hbm_base": 0x0, "ddr_base": 0x0}
name, target = "test", targets.hpp


def run_expr_check(dyn_arg, sta_arg, main_arg):
    def check(fn):
        name = fn.__name__
        main_list = []
        for k, v in main_arg.items():
            main_list.append(f"{k}:{v}")
        main_str = " ".join(main_list)
        name += " " + main_str
        start = f" RT Check Start: \033[36;34m{name}\033[36;0m "
        print(f"{start:=^135}")
        dyn_expr, ignores = fn(*dyn_arg)
        sta_expr, ignores = fn(*sta_arg)
        dyn_out = transform.infer_type(dyn_expr, device)
        sta_out = transform.infer_type(sta_expr, device)
        print(dyn_out)
        dyn_mod = backend.build(dyn_out, init_addr, name, False, target, configs)
        sta_mod = backend.build_tb(sta_out, init_addr, name, target, configs)
        dyn_rt = RuntimeBase(dyn_mod.lib)
        dyn_rt.main(name, **main_arg)
        sta_rt = RuntimeBase(sta_mod.lib)
        sta_rt.main(name, **main_arg)
        regs1 = dyn_rt.regs
        regs2 = sta_rt.regs
        # print(regs1)
        if RegsCheckSameList(regs1, regs2, ignores):
            success = f"\033[36;32m Check Success!\033[36;0m "
            finish = f" Check Finish: \033[36;32m{name}\033[36;0m "
            print(f"{success:-^135}")
        else:
            fail = f"\033[36;31m Check Fail! \033[36;0m"
            finish = f" RT Check Finish: \033[36;31m{name}\033[36;0m "
            print(f"{fail:*^135}")
        print(f"{finish:=^135}\n")
    return check


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def ln_compare(token, last_token):
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [2*chin], dtype=de.fp16)
    output = dlavm.op.nn.layer_norm(input, weight)
    return output, [130, 131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def rms_compare(token, last_token):
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [2*chin], dtype=de.fp16)
    output = dlavm.op.nn.rms_norm(input, weight)
    return output, [130, 131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def trp_mvm_compare(token, last_token):
    input_q = adr.var_hbm("input_q", [f_head, token, 128])
    input_k = adr.var_hbm("input_k", [w_head, token+last_token, 128])
    output = dlavm.op.nn.mvm_f16xf16(input_q, input_k, w_trp=True)
    return output, [10, 11, 13]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def f2w_mvm_compare(token, last_token):
    input_a = adr.var_hbm("input_a", [f_head, token, token+last_token])
    input_v = adr.var_hbm("input_v", [w_head, token+last_token, 128])
    output = dlavm.op.nn.mvm_f16xf16(input_a, input_v)
    return output, [10, 11, 13]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def kcache_compare(token, last_token):
    input_k = adr.var_hbm("input_k", [w_head, token, 128])
    output = dlavm.op.nn.kcache2hbm(input_k, cache_len=last_token)
    return output, [131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def vcache_compare(token, last_token):
    input_v = adr.var_hbm("input_v", [w_head, token, 128])
    output = dlavm.op.nn.vcache2hbm(input_v, cache_len=last_token)
    return output, [131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def mvm_compare(token, last_token):
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight", "test", [chout, chin])
    output = dlavm.op.nn.mvm_f16xi4(input, weight)
    return output, [10, 11, 13, 25]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def mvm_bn_compare(token, last_token, chin=13696, chout=4096):
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight", "test", [chout, chin])
    bn = adr.const_hbm("bn", "test", [2*chout], dtype=de.fp16)
    output = dlavm.op.nn.mvm_f16xi4(input, weight, bn)
    return output, [10, 11, 13, 25]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def mvm_bn_argmax_compare(token, last_token):
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight", "test", [chout, chin])
    bn = adr.const_hbm("bn", "test", [2*chout], dtype=de.fp16)
    output = dlavm.op.nn.mvm_f16xi4(input, weight, bn, argmax=True)
    return output, [10, 11, 13, 25]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def softmax_compare(token, last_token):
    input = adr.var_hbm("input", [f_head, token, token+last_token])
    output = dlavm.op.nn.softmax(input)
    return output, [131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def elementwise_compare(token, last_token):
    input1 = adr.var_hbm("input1", [1, token, chin])
    input2 = adr.var_hbm("input2", [1, token, chin])
    output = dlavm.op.nn.add(input1, input2)
    return output, [131, 134, 140]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def activate_compare(token, last_token):
    input = adr.var_hbm("input", [1, token, chin])
    weight = adr.const_hbm("weight1", "test", [chin], dtype=de.fp16)
    output = dlavm.op.nn.activate(input, weight)
    return output, [130, 131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def emb_glm_compare(token, last_token):
    input = adr.var_hbm("input", [f_head, token, 128])
    weight = adr.const_hbm("weight1", "test", [100, 128], dtype=de.fp16)
    output = dlavm.op.nn.rope_glm(input, weight, last_token=last_token)
    return output, [130, 131, 134]


@run_expr_check([token, last_token], [int_token, int_last_token], {str_token:int_token, str_last_token:int_last_token})
def emb_qwen_compare(token, last_token):
    input = adr.var_hbm("input", [f_head, token, 128])
    weight = adr.const_hbm("weight1", "test", [100, 128], dtype=de.fp16)
    output = dlavm.op.nn.rope_qwen(input, weight, last_token=last_token)
    return output, [130, 131, 134]