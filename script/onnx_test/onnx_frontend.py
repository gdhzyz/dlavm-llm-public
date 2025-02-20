import dlavm
from dlavm import adr
from dlavm import ne
from dlavm import codegen
from dlavm.transform import infer_type, offline_process, fused_ops
from dlavm import transform
import sys  # 导入sys模块


def chatglm_block(data, pos_weight, mask, silu_weight, token, index, arg_max):
    prefix = "BLOCK%02d_" % index
    ln_k_bias = adr.hbm.const_ddr(prefix + "ln_k_bias", None, [4096*2])
    qkv_weight = adr.hbm.const_hbm(prefix + "qkv_weight", None, [4096, 128*36])
    qkv_bn = adr.hbm.const_ddr(prefix + "qkv_bn_bias", None, [2*128*36])
    # k0_cache = adr.hbm.const_ddr(prefix + "k0_cache", None, [1, token-1, 128])
    # k1_cache = adr.hbm.const_ddr(prefix + "k1_cache", None, [1, token-1, 128])
    # v0_cache = adr.hbm.const_ddr(prefix + "v0_cache", None, [1, token-1, 128])
    # v1_cache = adr.hbm.const_ddr(prefix + "v1_cache", None, [1, token-1, 128])
    rsqrt = adr.hbm.const_ddr(prefix + "rsqrt", None, [token*2])
    atten_weight = adr.hbm.const_hbm(prefix + "atten_weight", None, [4096, 4096])
    atten_bias = adr.hbm.const_ddr(prefix + "atten_bn", None, [4096*2])
    post_k_bias = adr.hbm.const_ddr(prefix + "post_k_bias", None, [4096*2])
    h_to_4h_wt_0 = adr.hbm.const_hbm(prefix + "h_to_4h_wt_0", None, [4096, 13696])
    h_to_4h_wt_1 = adr.hbm.const_hbm(prefix + "h_to_4h_wt_1", None, [4096, 13696])
    h_to_4h_bn_0 = adr.hbm.const_ddr(prefix + "h_to_4h_bn_0", None, [13696*2])
    h_to_4h_bn_1 = adr.hbm.const_ddr(prefix + "h_to_4h_bn_1", None, [13696*2])
    dense_4h_to_4h_wt = adr.hbm.const_hbm(prefix + "dense_4h_to_h_wt", None, [13696, 4096])
    dense_4h_to_4h_bn = adr.hbm.const_ddr(prefix + "dense_4h_to_h_bn", None, [4096*2])

    ln_out = adr.hbm.layer_norm(data, ln_k_bias)
    qkv_data = adr.hbm.mvm_bn(ln_out, qkv_weight, qkv_bn)
    qkv_data = adr.split(qkv_data, 2, [4096, 128, 128, 128, 128])
    q_data = adr.reshape(qkv_data[0], [32, token, 128])
    q_data = adr.hbm.pos_emb(q_data, pos_weight)
    k_data0 = adr.hbm.pos_emb(qkv_data[1], pos_weight)
    k_data1 = adr.hbm.pos_emb(qkv_data[2], pos_weight)
    # k_data0 = adr.cpu.cache(k_data0, k0_cache)
    # k_data1 = adr.cpu.cache(k_data1, k1_cache)
    k_data0 = adr.hbm.transpose(k_data0)
    k_data1 = adr.hbm.transpose(k_data1)
    scores = adr.hbm.mvm_bn_res(q_data, k_data0, k_data1, rsqrt, mask, skip=2)
    scores = adr.hbm.softmax(scores)
    v_data0 = qkv_data[3]
    v_data1 = qkv_data[4]
    # v_data0 = adr.cpu.cache(qkv_data[3], v0_cache)
    # v_data1 = adr.cpu.cache(qkv_data[4], v1_cache)
    v_data0 = adr.hbm.feature2weight(v_data0)
    v_data1 = adr.hbm.feature2weight(v_data1)
    atten_out = adr.hbm.mvm(scores, v_data0, v_data1, skip=2)
    atten_out = adr.reshape(atten_out, [1, token, 4096])
    res_out = adr.hbm.mvm_bn_res(atten_out, atten_weight, atten_bias, data)
    post_atten = adr.hbm.layer_norm(res_out, post_k_bias)
    dense_4h_out0 = adr.hbm.mvm_bn(post_atten, h_to_4h_wt_0, h_to_4h_bn_0)
    act_output = adr.hbm.activate(dense_4h_out0, silu_weight)
    dense_4h_out = adr.hbm.mvm_bn_res(post_atten, h_to_4h_wt_1, h_to_4h_bn_1, act_output, res_mul=1)
    output = adr.hbm.mvm_bn_res(dense_4h_out, dense_4h_to_4h_wt, dense_4h_to_4h_bn, res_out, arg_max=arg_max)
    return output


def chatglm_without_kvcache():
    sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
    token = ne.Var("token", 19)
    data = adr.hbm.var_ddr("data", [1, token, 4096])
    pos_weight = adr.hbm.const_ddr("pos_emb", None, [1, 128, 64])
    mask = adr.hbm.const_ddr("mask", None, [32, token, token])
    silu_weight = adr.hbm.const_ddr("silu_act", None, [32*3], adr.DataEnum.int8)
    block_size = 1
    for n in range(block_size):
        data = chatglm_block(data, pos_weight, mask, silu_weight, token, n, n==block_size-1)
    output = data[1]

    output = infer_type(output, dlavm.device.HBM0321)
    from dlavm.driver import config
    # config.tb_sim_path = "/home/previous/accel/hbm0227/driver/HBM_sv"
    expr, source, storage, _ = codegen.csb_head(output, "chatglm", 0x200000000, 0x0)
    with open("./test/source.h", "w") as f:
        f.write(source)
    print(expr)
    print(storage)
    

def check_testbench():
    from dlavm.driver import config
    config.tb_sim_path = "/home/previous/accel/hbm0201/driver/HBM_sv"
    data = adr.hbm.var_ddr("data", [1, 1, 4096])
    weight = adr.hbm.const_hbm("weight", None, [4096, 128*36])
    output = adr.hbm.mvm(data, weight)
    print(output)

    output = infer_type(output)
    expr, source, storage, mod = codegen.testbench(output, "chatglm", 0x200000000, 0x0)
    expr, source, storage, mod1 = codegen.csb_head(output, "chatglm", 0x200000000, 0x0)
    with open("./test/source.h", "w") as f:
        f.write(source)
    print(expr)
    print(storage)
    print(mod)
    print(mod1)


def mp_booth():
    data = adr.booth.var_ddr("data", [3, 3, 3])
    weight = adr.booth.const_ddr("weight", None, [8, 3, 3, 3])
    output = adr.booth.conv2d(data, weight, [1, 1], [0, 0], 8, 8, [2, 3, 1])

    output = infer_type(output)
    expr, source, storage, mod1 = codegen.csb_head(output, "conv", 0x40000000, 0x0)
    with open("./test/source.h", "w") as f:
        f.write(source)
    print(expr)
    print(storage)


import onnx
from dlavm import frontend

def onnx_test():
    mod = onnx.load("./test/glm2_block_test.onnx")
    expr = frontend.from_onnx(mod)
    print(expr)

    output = infer_type(expr)
    print(output)
    expr, source, storage, mod1 = codegen.csb_head(output, "test", 0x200000000, 0x0)
    with open("./test/test.h", "w") as f:
        f.write(source)
    print(storage)


def test():
    token = 19
    data = adr.hbm.var_ddr("data", [2, token, 128])
    pos_weight = adr.hbm.const_ddr("pos_emb", None, [1, 128, 64])
    output = adr.hbm.pos_emb(data, pos_weight)

    output = infer_type(output)
    from dlavm.driver import config
    config.tb_sim_path = "/home/previous/accel/hbm0227/driver/HBM_sv"
    expr, source, storage, mod_0 = codegen.csb_test_head(output, "chatglm", 0x200000000, 0x0)
    _, _, _, mod_1 = codegen.testbench_test_head(output, "chatglm", 0x200000000, 0x0)
    print(mod_0 == mod_1)
    with open("./test/POS_EMB_2x19x128.h", "w") as f:
        f.write(source)
    print(expr)
    print(storage)


def test_mvm():
    token = 19
    data = adr.hbm.var_ddr("data", [1, token, 4096])
    weight = adr.hbm.const_hbm("weight", None, [4096, 128])
    bn = adr.hbm.const_ddr("bn", None, [2*128])
    output = adr.hbm.mvm_bn(data, weight, bn)

    output = infer_type(output)
    from dlavm.driver import config
    config.tb_sim_path = "/home/previous/accel/hbm0227/driver/HBM_sv"
    expr, source, storage, mod_0 = codegen.csb_test_head(output, "chatglm", 0x200000000, 0x0)
    _, _, _, mod_1 = codegen.testbench_test_head(output, "chatglm", 0x200000000, 0x0)
    print(mod_0 == mod_1)
    with open("./test/MVMBN_4096x128.h", "w") as f:
        f.write(source)
    print(expr)
    print(storage)



def load_onnx_block():
    mod = onnx.load("./test/glm2_block.onnx")
    # sys.setrecursionlimit(1000)
    expr = frontend.from_onnx(mod)
    print(expr)

    graph_opt = transform.Sequential([
        transform.FusedOps(), # fused mvm with bias
        transform.EliminateReshape(),
        transform.FusedOps(opt_level=2), # fused mvm or mvm_bn with res
        transform.OfflineProcess(),
    ])
    output = graph_opt(expr)
    print(output)
    expr, source, storage, mod1 = codegen.csb_head(output, "test", 0x200000000, 0x0)
    with open("./test/test.h", "w") as f:
        f.write(source)
    print(storage)


if __name__ == "__main__":
    a = 12 * (ne.Var("test") + 2)
    b = 12 * (ne.Numb(10) + 2)
    a = a.simplify()
    b = b.simplify()
    print("a:", a.export("cpp"))
    print("b:", b.export("cpp"))
    # load_onnx_block()
    # mp_booth()
    # chatglm_without_kvcache()
    # check_testbench()
    # onnx_test()
    # test()
