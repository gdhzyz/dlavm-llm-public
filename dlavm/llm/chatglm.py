import dlavm
from dlavm import adr
from dlavm import ne
import sys
import dlavm.utils
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

'''
KVCache with HBM Storage Version
'''
def chatglm_block_hbm(data, pos_weight, silu_weight, token, index, path_prefix="BLOCK_write_data", ir=False):
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

    if ir:
        k_data = adr.hbm.dat_hbm(k_data, 1)
        v_data = adr.hbm.dat_hbm(v_data, 0)
    else:
        k_data = adr.hbm.dat2hbm(k_data, 1)
        v_data = adr.hbm.dat2hbm(v_data, 0)
    atten_out = adr.hbm.trp_mvm(q_data, k_data)
    atten_out = adr.hbm.softmax(atten_out)
    atten_out = adr.hbm.f2w_mvm(atten_out, v_data)
    atten_out = adr.reshape(atten_out, [1, token, 4096])

    res_out = adr.hbm.mvm_bn_res(atten_out, atten_weight, atten_bias, data)
    post_atten = adr.hbm.rms_norm(res_out, post_k_bias)
    dense_4h_out0 = adr.hbm.mvm_bn(post_atten, h_to_4h_wt_0, h_to_4h_bn_0)
    act_output = adr.hbm.activate(dense_4h_out0, silu_weight)
    dense_4h_out = adr.hbm.mvm_bn_res(post_atten, h_to_4h_wt_1, h_to_4h_bn_1, act_output, res_mul=1)
    output = adr.hbm.mvm_bn_res(dense_4h_out, dense_4h_to_4h_wt, dense_4h_to_4h_bn, res_out)
    return output


def chatglm_expr_hbm(device, path_prefix, debug, ir=False):
    block_size = 1 if debug else 28
    token = ne.Var("seq", device.MAX_TOKEN) if ir else ne.Var("token", device.MAX_TOKEN)
    data = adr.hbm.var_ddr("data_in", [1, token, 4096])
    pos_weight = adr.hbm.const_ddr("pos_emb", path_prefix + f"/pos_in_4_token{device.MAX_TOKEN}_ch32.bin", [1, device.MAX_TOKEN*2, 64])
    pos_weight.prefix = "global"
    silu_weight = adr.hbm.const_ddr("silu_act", path_prefix + "/ACT_parameter_in_DDR.bin", [32*128], adr.DataEnum.fp16)
    silu_weight.prefix = "global"
    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    for n in range(block_size):
        data = chatglm_block_hbm(data, pos_weight, silu_weight, token, n, path_prefix=path_prefix, ir=ir)
    ln_k_bias = adr.hbm.const_ddr("Final_LN_k_bias", path_prefix + "/OutLayer/LN_DDR_bin/LN_wt_in_DDR.bin", [4096*2])
    ln_out = adr.hbm.rms_norm(data, ln_k_bias, kvcache=1, kvcache_offset=ne.If(kvcache, 0, 1), last_token=last_token, kvcache_token=True)
    output_wt = adr.hbm.const_hbm("Output_Layer_wt", path_prefix + "/OutLayer/MVM_BN_write_to_HBM_bin/MVMBN_Argmax_HBM_DDR_%02d.bin", [4096, 65024])
    output_bn = adr.hbm.const_ddr("Output_Layer_bn", path_prefix + "/OutLayer/MVM_BN_DDR_bin/MVMBN_Argmax_wt_and_bias_in_DDR.bin", [65024*2])
    mvm_out = adr.hbm.mvm_bn(ln_out, output_wt, output_bn, arg_max=1, kvcache=1)
    output = mvm_out[1]
    return output


'''
KVCache with DDR Storage Version
'''
def chatglm_block(data, pos_weight, silu_weight, token, index, path_prefix="BLOCK_write_data"):
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
    v_data = adr.hbm.cache(v_data)
    q_data = adr.reshape(q_data, [32, token, 128])
    k_data = adr.reshape(k_data, [2, token, 128])
    v_data = adr.reshape(v_data, [2, token, 128])

    q_data = adr.hbm.pos_emb(q_data, pos_weight)
    k_data = adr.hbm.pos_emb(k_data, pos_weight)
    k_data = adr.hbm.cache(k_data)
    atten_out = adr.hbm.mvm_afterTRP(q_data, k_data)
    atten_out = adr.hbm.softmax(atten_out)
    atten_out = adr.hbm.mvm_afterF2W(atten_out, v_data)
    atten_out = adr.reshape(atten_out, [1, token, 4096])

    res_out = adr.hbm.mvm_bn_res(atten_out, atten_weight, atten_bias, data)
    post_atten = adr.hbm.rms_norm(res_out, post_k_bias)
    dense_4h_out0 = adr.hbm.mvm_bn(post_atten, h_to_4h_wt_0, h_to_4h_bn_0)
    act_output = adr.hbm.activate(dense_4h_out0, silu_weight)
    dense_4h_out = adr.hbm.mvm_bn_res(post_atten, h_to_4h_wt_1, h_to_4h_bn_1, act_output, res_mul=1)
    output = adr.hbm.mvm_bn_res(dense_4h_out, dense_4h_to_4h_wt, dense_4h_to_4h_bn, res_out)
    return output

def chatglm_expr(device, path_prefix, debug):
    block_size = 1 if debug else 28
    token = ne.Var("token", device.MAX_TOKEN)
    data = adr.hbm.var_ddr("data_in", [1, token, 4096])
    pos_weight = adr.hbm.const_ddr("pos_emb", path_prefix + f"/pos_in_4_token{device.MAX_TOKEN}_ch32.bin", [1, device.MAX_TOKEN*2, 64])
    pos_weight.prefix = "global"
    silu_weight = adr.hbm.const_ddr("silu_act", path_prefix + "/ACT_parameter_in_DDR.bin", [32*128], adr.DataEnum.fp16)
    silu_weight.prefix = "global"
    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    for n in range(block_size):
        data = chatglm_block(data, pos_weight, silu_weight, token, n, path_prefix=path_prefix)
    ln_k_bias = adr.hbm.const_ddr("Final_LN_k_bias", path_prefix + "/OutLayer/LN_DDR_bin/LN_wt_in_DDR.bin", [4096*2])
    ln_out = adr.hbm.rms_norm(data, ln_k_bias, kvcache=1, kvcache_offset=ne.If(kvcache, 0, 1), last_token=last_token)
    output_wt = adr.hbm.const_hbm("Output_Layer_wt", path_prefix + "/OutLayer/MVM_BN_write_to_HBM_bin/MVMBN_Argmax_HBM_DDR_%02d.bin", [4096, 65024])
    output_bn = adr.hbm.const_ddr("Output_Layer_bn", path_prefix + "/OutLayer/MVM_BN_DDR_bin/MVMBN_Argmax_wt_and_bias_in_DDR.bin", [65024*2])
    mvm_out = adr.hbm.mvm_bn(ln_out, output_wt, output_bn, arg_max=1, kvcache=1)
    output = mvm_out[1]
    return output

