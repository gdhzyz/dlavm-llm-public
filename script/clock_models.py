import os
import dlavm
from dlavm import adr
from dlavm import ne
from dlavm import codegen
from dlavm import transform
import sys
import json
import argparse
import numpy as np
from time import strftime, localtime

import dlavm.utils


def chatglm_block_dynamic(data, pos_weight, silu_weight, token, index, kvcache, last_token, path_prefix="BLOCK_write_data"):
    prefix = "BLOCK%02d_" % index
    data_path = path_prefix + "/BLOCK%02d/" % index
    ln_k_bias = adr.hbm.const_ddr(prefix + "ln_k_bias", data_path + "LN_DDR_bin/LN0_wt_in_DDR.bin", [4096*2])
    qk_weight = adr.hbm.const_hbm(prefix + "q_weight", data_path + "MVM_BN_write_to_HBM_bin/MVMBN0_q_HBM_DDR_%02d.bin", [4096, 128*34])
    qk_bn = adr.hbm.const_ddr(prefix + "q_bn_bias", data_path + "MVM_BN_DDR_bin/MVMBN0_q_wt_and_bias_in_DDR.bin", [2*128*34])
    v_weight = adr.hbm.const_hbm(prefix + "v_weight", data_path + "MVM_BN_write_to_HBM_bin/MVMBN0_v_HBM_DDR_%02d.bin", [4096, 128*2])
    v_bn = adr.hbm.const_ddr(prefix + "v_bn_bias", data_path + "MVM_BN_DDR_bin/MVMBN0_v_wt_and_bias_in_DDR.bin", [2*128*2])
    atten_weight = adr.hbm.const_hbm(prefix + "atten_weight", data_path + "MVM_BN_RES_write_to_HBM_bin/MVMBNRES0_HBM_DDR_%02d.bin", [4096, 4096])
    atten_bias = adr.hbm.const_ddr(prefix + "atten_bn", data_path + "MVM_BN_RES_DDR_bin/MVMBNRES0_wt_and_bias_in_DDR.bin", [4096*2])
    post_k_bias = adr.hbm.const_ddr(prefix + "post_k_bias", data_path + "LN_DDR_bin/LN1_wt_in_DDR.bin", [4096*2])
    h_to_4h_wt_0 = adr.hbm.const_hbm(prefix + "h_to_4h_wt_0", data_path + "MVM_BN_write_to_HBM_bin/MVMBN1_HBM_DDR_%02d.bin", [4096, 13696])
    h_to_4h_wt_1 = adr.hbm.const_hbm(prefix + "h_to_4h_wt_1", data_path + "MVM_BN_RES_write_to_HBM_bin/MVMBNRES1_HBM_DDR_%02d.bin", [4096, 13696])
    h_to_4h_bn_0 = adr.hbm.const_ddr(prefix + "h_to_4h_bn_0", data_path + "MVM_BN_DDR_bin/MVMBN1_wt_and_bias_in_DDR.bin", [13696*2])
    h_to_4h_bn_1 = adr.hbm.const_ddr(prefix + "h_to_4h_bn_1", data_path + "MVM_BN_RES_DDR_bin/MVMBNRES1_wt_and_bias_in_DDR.bin", [13696*2])
    dense_4h_to_4h_wt = adr.hbm.const_hbm(prefix + "dense_4h_to_h_wt", data_path + "MVM_BN_RES_write_to_HBM_bin/MVMBNRES2_HBM_DDR_%02d.bin", [13696, 4096])
    dense_4h_to_4h_bn = adr.hbm.const_ddr(prefix + "dense_4h_to_h_bn", data_path + "MVM_BN_RES_DDR_bin/MVMBNRES2_wt_and_bias_in_DDR.bin", [4096*2])

    ln_out = adr.hbm.layer_norm(data, ln_k_bias, rms=1, kvcache=kvcache, last_token=last_token)

    qk_data = adr.hbm.mvm_bn(ln_out, qk_weight, qk_bn, kvcache=kvcache, last_token=last_token)
    v_data = adr.hbm.mvm_bn(ln_out, v_weight, v_bn, kvcache=kvcache, last_token=last_token)
    v_data = adr.hbm.cache(v_data)
    qk_data = adr.split(adr.reshape(qk_data, [34, token, 128]), axis=0, new_chs=[32, 2])
    q_data = qk_data[0]
    k_data = qk_data[1]
    v_data = adr.reshape(v_data, [2, token, 128])

    q_data = adr.hbm.pos_emb(q_data, pos_weight, kvcache=kvcache, last_token=last_token)
    k_data = adr.hbm.pos_emb(k_data, pos_weight, kvcache=kvcache, last_token=last_token)
    k_data = adr.hbm.cache(k_data)
    atten_out = adr.hbm.mvm_afterTRP(q_data, k_data, kvcache=kvcache, last_token=last_token)
    atten_out = adr.hbm.softmax(atten_out, kvcache=kvcache, last_token=last_token)
    atten_out = adr.hbm.mvm_afterF2W(atten_out, v_data, kvcache=kvcache, last_token=last_token)
    atten_out = adr.reshape(atten_out, [1, token, 4096])

    res_out = adr.hbm.mvm_bn_res(atten_out, atten_weight, atten_bias, data, kvcache=kvcache, last_token=last_token)
    post_atten = adr.hbm.layer_norm(res_out, post_k_bias, rms=1, kvcache=kvcache, last_token=last_token)
    dense_4h_out0 = adr.hbm.mvm_bn(post_atten, h_to_4h_wt_0, h_to_4h_bn_0, kvcache=kvcache, last_token=last_token)
    act_output = adr.hbm.activate(dense_4h_out0, silu_weight, kvcache=kvcache, last_token=last_token)
    dense_4h_out = adr.hbm.mvm_bn_res(post_atten, h_to_4h_wt_1, h_to_4h_bn_1, act_output, res_mul=1, kvcache=kvcache, last_token=last_token)
    output = adr.hbm.mvm_bn_res(dense_4h_out, dense_4h_to_4h_wt, dense_4h_to_4h_bn, res_out, kvcache=kvcache, last_token=last_token)
    return output


def chatglm_main_dynamic(device=dlavm.device.HBM0321, clock=1, path_prefix = "BLOCK_write_data"):
    block_size = 1 if clock else block_size
    token = ne.Var("token", device.MAX_TOKEN)
    name = f"chatglm_{device.MAX_TOKEN}"
    name += "_clock" if clock else ""
    name += "_" + strftime('%m%d_%H%M', localtime())
    data = adr.hbm.var_ddr("data_in", [1, token, 4096])
    pos_weight = adr.hbm.const_ddr("pos_emb", path_prefix + f"/pos_in_4_token{device.MAX_TOKEN}_ch32.bin", [1, device.MAX_TOKEN*2, 64])
    pos_weight.prefix = "global"
    silu_weight = adr.hbm.const_ddr("silu_act", path_prefix + "/ACT_parameter_in_DDR.bin", [32*128], adr.DataEnum.fp16)
    silu_weight.prefix = "global"
    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    data = chatglm_block_dynamic(data, pos_weight, silu_weight, token, 0, kvcache=kvcache, last_token=last_token, path_prefix=path_prefix)
    ln_k_bias = adr.hbm.const_ddr("Final_LN_k_bias", path_prefix + "/OutLayer/LN_DDR_bin/LN_wt_in_DDR.bin", [4096*2])
    ln_out = adr.hbm.layer_norm(data, ln_k_bias, rms=1, kvcache=1, kvcache_offset=ne.If(kvcache, 0, 1), last_token=last_token)
    output_wt = adr.hbm.const_hbm("Output_Layer_wt", path_prefix + "/OutLayer/MVM_BN_write_to_HBM_bin/MVMBN_Argmax_HBM_DDR_%02d.bin", [4096, 65024])
    output_bn = adr.hbm.const_ddr("Output_Layer_bn", path_prefix + "/OutLayer/MVM_BN_DDR_bin/MVMBN_Argmax_wt_and_bias_in_DDR.bin", [65024*2])
    mvm_out = adr.hbm.mvm_bn(ln_out, output_wt, output_bn, arg_max=1, kvcache=1)
    output = mvm_out[1]

    output = transform.infer_type(output, device, attrs={"full":0})
    expr, source, storage, _ = codegen.csb_test_clock_ops(output, name, {"global": 0x200000000, "weight": "global", "cache": "weight", "runtime": "cache", "hbm": 0x0})
    save_path = os.path.join("output", name + ".h")
    log_path = os.path.join("output", name + ".log")
    with open(save_path, "w") as f:
        f.write(source)
    dlavm.utils.LOG_WITH_PREFIX("expression", str(expr))
    dlavm.utils.LOG_WITH_PREFIX("storage", str(storage))
    dlavm.utils.LOG_EXPORT(log_path)
    print(save_path)
    print(log_path)


def llama_block_dynamic(data, pos_weight, silu_weight, token, index, kvcache, last_token, path_prefix="BLOCK_write_data"):
    prefix = "BLOCK%02d_" % index
    data_path = path_prefix + "/BLOCK%02d/" % index
    ln_k_bias = adr.hbm.const_ddr(prefix + "ln_k_bias", data_path + "LN_DDR_bin/LN0_wt_in_DDR.bin", [4096*2])
    qk_weight = adr.hbm.const_hbm(prefix + "qk_weight", data_path + "MVM_BN_write_to_HBM_bin/MVMBN0_qk_HBM_DDR_%02d.bin", [4096, 128*32*2])
    v_weight = adr.hbm.const_hbm(prefix + "v_weight", data_path + "MVM_BN_write_to_HBM_bin/MVMBN0_v_HBM_DDR_%02d.bin", [4096, 128*32])
    atten_weight = adr.hbm.const_hbm(prefix + "atten_weight", data_path + "MVM_BN_RES_write_to_HBM_bin/MVMBNRES0_HBM_DDR_%02d.bin", [4096, 4096])
    post_k_bias = adr.hbm.const_ddr(prefix + "post_k_bias", data_path + "LN_DDR_bin/LN1_wt_in_DDR.bin", [4096*2])
    h_to_4h_wt_0 = adr.hbm.const_hbm(prefix + "h_to_4h_wt_0", data_path + "MVM_BN_write_to_HBM_bin/MVMBN1_HBM_DDR_%02d.bin", [4096, 11008])
    h_to_4h_wt_1 = adr.hbm.const_hbm(prefix + "h_to_4h_wt_1", data_path + "MVM_BN_RES_write_to_HBM_bin/MVMBNRES1_HBM_DDR_%02d.bin", [4096, 11008])
    h_to_4h_bn_1 = adr.hbm.const_ddr(prefix + "h_to_4h_bn_1", data_path + "MVM_BN_RES_DDR_bin/MVMBNRES1_wt_and_bias_in_DDR.bin", [11008*2])
    dense_4h_to_4h_wt = adr.hbm.const_hbm(prefix + "dense_4h_to_h_wt", data_path + "MVM_BN_RES_write_to_HBM_bin/MVMBNRES2_HBM_DDR_%02d.bin", [11008, 4096])
    bn_zeros = adr.hbm.const_ddr(prefix + "bn_zeros", data_path + "MVM_BN_RES_DDR_bin/DDR_BN_zeros.bin", [4096*2])

    ln_out = adr.hbm.layer_norm(data, ln_k_bias, rms=1, kvcache=kvcache, last_token=last_token)

    qk_data = adr.hbm.mvm(ln_out, qk_weight, kvcache=kvcache, last_token=last_token)
    v_data = adr.hbm.mvm(ln_out, v_weight, kvcache=kvcache, last_token=last_token)
    v_data = adr.hbm.cache(v_data)
    qk_data = adr.reshape(qk_data, [32*2, token, 128])
    qk_data = adr.split(qk_data, 0, [32, 32])
    q_data, k_data = qk_data[0], qk_data[1]
    v_data = adr.reshape(v_data, [32, token, 128])

    q_data = adr.hbm.pos_emb(q_data, pos_weight, kvcache=kvcache, last_token=last_token)
    k_data = adr.hbm.pos_emb(k_data, pos_weight, kvcache=kvcache, last_token=last_token)
    k_data = adr.hbm.cache(k_data)
    atten_out = adr.hbm.mvm_afterTRP(q_data, k_data, kvcache=kvcache, last_token=last_token)
    atten_out = adr.hbm.softmax(atten_out, kvcache=kvcache, last_token=last_token)
    atten_out = adr.hbm.mvm_afterF2W(atten_out, v_data, kvcache=kvcache, last_token=last_token)
    atten_out = adr.reshape(atten_out, [1, token, 4096])

    res_out = adr.hbm.mvm_bn_res(atten_out, atten_weight, bn_zeros, data, kvcache=kvcache, last_token=last_token)
    post_atten = adr.hbm.layer_norm(res_out, post_k_bias, rms=1, kvcache=kvcache, last_token=last_token)
    dense_4h_out0 = adr.hbm.mvm(post_atten, h_to_4h_wt_0, kvcache=kvcache, last_token=last_token)
    act_output = adr.hbm.activate(dense_4h_out0, silu_weight, kvcache=kvcache, last_token=last_token)
    dense_4h_out = adr.hbm.mvm_bn_res(post_atten, h_to_4h_wt_1, h_to_4h_bn_1, act_output, res_mul=1, kvcache=kvcache, last_token=last_token)
    output = adr.hbm.mvm_bn_res(dense_4h_out, dense_4h_to_4h_wt, bn_zeros, res_out, kvcache=kvcache, last_token=last_token)
    return output


def llama_main_dynamic(device=dlavm.device.HBM0321, clock=0, path_prefix = "BLOCK_write_data"):
    block_size = 1
    sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
    token = ne.Var("token", device.MAX_TOKEN)
    name = f"llama_{device.MAX_TOKEN}"
    name += "_clock" if clock else ""
    name += "_" + strftime('%m%d_%H%M', localtime())
    data = adr.hbm.var_ddr("data_in", [1, token, 4096])
    pos_weight = adr.hbm.const_ddr("pos_emb", path_prefix + f"/pos_in_4_token{device.MAX_TOKEN}_ch32.bin", [1, device.MAX_TOKEN*2, 64])
    pos_weight.prefix = "global"
    silu_weight = adr.hbm.const_ddr("silu_act", path_prefix + "/ACT_parameter_in_DDR.bin", [32*128], adr.DataEnum.fp16)
    silu_weight.prefix = "global"
    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    for n in range(block_size):
        data = llama_block_dynamic(data, pos_weight, silu_weight, token, n, kvcache=kvcache, last_token=last_token, path_prefix=path_prefix)
    ln_k_bias = adr.hbm.const_ddr("Final_LN_k_bias", path_prefix + "/OutLayer/LN_DDR_bin/LN_wt_in_DDR.bin", [4096*2])
    ln_out = adr.hbm.layer_norm(data, ln_k_bias, rms=1, kvcache=1, kvcache_offset=ne.If(kvcache, 0, 1), last_token=last_token)
    output_wt = adr.hbm.const_hbm("Output_Layer_wt", path_prefix + "/OutLayer/MVM_BN_write_to_HBM_bin/MVMBN_Argmax_HBM_DDR_%02d.bin", [4096, 32000])
    output_bn = adr.hbm.const_ddr("Output_Layer_bn", path_prefix + "/OutLayer/MVM_BN_DDR_bin/MVMBN_Argmax_wt_and_bias_in_DDR.bin", [32000*2])
    mvm_out = adr.hbm.mvm_bn(ln_out, output_wt, output_bn, arg_max=1, kvcache=1)
    output = mvm_out[1]

    output = transform.infer_type(output, device, attrs={"full":0})
    expr, source, storage, _ = codegen.csb_test_clock_ops(output, name, {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "hbm": 0x0})
    save_path = os.path.join("output", name + ".h")
    log_path = os.path.join("output", name + ".log")
    with open(save_path, "w") as f:
        f.write(source)
    dlavm.utils.LOG_WITH_PREFIX("expression", str(expr))
    dlavm.utils.LOG_WITH_PREFIX("storage", str(storage))
    dlavm.utils.LOG_EXPORT(log_path)
    print(save_path)
    print(log_path)


if __name__ == "__main__":
    device = dlavm.device.EdgeLLMv3

    parser = argparse.ArgumentParser()
    parser.add_argument("--clock", action="store_true", default=True, help="clock backend mode")
    parser.add_argument("--prefix", type=str, default="BLOCK_write_data", help="set datapath prefix")
    parser.add_argument("--maxtoken", type=int, default=device.MAX_TOKEN, help="set max token")
    args = parser.parse_args()

    device.MAX_TOKEN = args.maxtoken

    chatglm_main_dynamic(device=device, clock=args.clock, path_prefix=args.prefix)
    dlavm.utils.LOG_CLEAR()
    llama_main_dynamic(device=device, clock=args.clock, path_prefix=args.prefix)

