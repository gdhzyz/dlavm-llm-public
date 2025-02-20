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
import dlavm.device
import dlavm.utils
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000


def expression(data, token, pos_weight, silu_weight, path_prefix, index):
    prefix = "BLOCK%02d_" % index
    data_path = path_prefix + "/BLOCK%02d/" % index
    ln_k_bias = adr.hbm.const_ddr(prefix + "ln_k_bias", data_path + "LN_DDR_bin/LN0_wt_in_DDR.bin", [4096*2])
    q_weight = adr.hbm.const_hbm(prefix + "q_weight", data_path + "MVM_BN_write_to_HBM_bin/MVMBN0_q_HBM_DDR_%02d.bin", [4096, 128*32])
    q_bn = adr.hbm.const_ddr(prefix + "q_bn_bias", data_path + "MVM_BN_DDR_bin/MVMBN0_q_wt_and_bias_in_DDR.bin", [2*128*32])
    k_weight = adr.hbm.const_hbm(prefix + "k_weight", data_path + "MVM_BN_write_to_HBM_bin/MVMBN0_k_HBM_DDR_%02d.bin", [4096, 128*2])
    k_bn = adr.hbm.const_ddr(prefix + "k_bn_bias", data_path + "MVM_BN_DDR_bin/MVMBN0_k_wt_and_bias_in_DDR.bin", [2*128*2])
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

    ln_out = adr.hbm.rms_norm(data, ln_k_bias)
    q_data = adr.hbm.mvm_bn(ln_out, q_weight, q_bn)
    k_data = adr.hbm.mvm_bn(ln_out, k_weight, k_bn)
    v_data = adr.hbm.mvm_bn(ln_out, v_weight, v_bn)
    q_data = adr.reshape(q_data, [32, token, 128])
    k_data = adr.reshape(k_data, [2, token, 128])
    v_data = adr.reshape(v_data, [2, token, 128])

    q_data = adr.hbm.pos_emb(q_data, pos_weight)
    k_data = adr.hbm.pos_emb(k_data, pos_weight)
    k_data = adr.hbm.dat2hbm(k_data, 1)
    atten_out = adr.hbm.trp_mvm(q_data, k_data)
    atten_out = adr.hbm.softmax(atten_out)
    v_data = adr.hbm.dat2hbm(v_data, 0)
    atten_out = adr.hbm.f2w_mvm(atten_out, v_data)
    atten_out = adr.reshape(atten_out, [1, token, 4096])

    res_out = adr.hbm.mvm_bn_res(atten_out, atten_weight, atten_bias, data)
    post_atten = adr.hbm.rms_norm(res_out, post_k_bias)
    dense_4h_out0 = adr.hbm.mvm_bn(post_atten, h_to_4h_wt_0, h_to_4h_bn_0)
    act_output = adr.hbm.activate(dense_4h_out0, silu_weight)
    dense_4h_out = adr.hbm.mvm_bn_res(post_atten, h_to_4h_wt_1, h_to_4h_bn_1, act_output, res_mul=1)
    output = adr.hbm.mvm_bn_res(dense_4h_out, dense_4h_to_4h_wt, dense_4h_to_4h_bn, res_out)
    return output


def compile(device, path_prefix="/home/sustech/Desktop/bin_file"):
    token = ne.Var("token", device.MAX_TOKEN)
    data = adr.hbm.var_ddr("data_in", [1, token, 4096])
    pos_weight = adr.hbm.const_ddr("pos_emb", path_prefix + f"/pos_in_4_token{device.MAX_TOKEN}_ch32.bin", [1, device.MAX_TOKEN*2, 64])
    pos_weight.prefix = "global"
    silu_weight = adr.hbm.const_ddr("silu_act", path_prefix + "/ACT_parameter_in_DDR.bin", [32*128], adr.DataEnum.fp16)
    silu_weight.prefix = "global"
    output = expression(data, token, pos_weight, silu_weight, path_prefix, 0)
    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    global_attrs = {"full":0, "kvcache": kvcache, "last_token": last_token}
    output = transform.infer_type(output, device, attrs=global_attrs)

    addr_assign_aux = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "cfg": "runtime", "hbm": 0x0, "onchip": 0x0}
    addr_assign = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "hbm": 0x0, "hbm_cache": "hbm"}
    expr, mod, storage = codegen.GraphCSBHead().build(output, addr_assign)
    source = codegen.CodeGenV80Head().build("v80_test", mod, storage, expr.get_device())

    _, prototxt, _, _ = codegen.visualize_prototxt(output, "v80_test", addr_assign, False)

    save_path = os.path.join("output", "v80_test.h")
    prototxt_path = os.path.join("output", "v80_test.prototxt")
    log_path = os.path.join("output", "v80_test.log")
    with open(save_path, "w") as f:
        f.write(source)
    with open(prototxt_path, "w") as f:
        f.write(prototxt)
    dlavm.utils.LOG_WITH_PREFIX("expression", str(expr))
    dlavm.utils.LOG_WITH_PREFIX("storage", str(storage))
    dlavm.utils.LOG_EXPORT(log_path)
    print(save_path)
    print(log_path)


compile(device=dlavm.device.HBM0912)