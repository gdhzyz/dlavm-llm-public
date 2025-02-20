import os
import torch
import dlavm
import numpy as np
from dlavm import ne
from dlavm import adr
from dlavm import transform
from dlavm import codegen
import quant

np.random.seed(0)


def quant_sim(model):
    qmodel = quant.QuantLinear(bits=4, groupsize=8, infeatures=8, outfeatures=1, bias=False)
    # qmodel.qweight = model['model.layers.0.self_attn.q_proj.qweight'][:1, :1]
    qmodel.qweight = torch.Tensor([[0]]).to(torch.int32)
    # qmodel.scales = model['model.layers.0.self_attn.q_proj.scales'][:1, :1]
    qmodel.scales = torch.Tensor([[0]]).to(torch.float16)
    # qmodel.qzeros = model['model.layers.0.self_attn.q_proj.qzeros'][:1, :1]
    qmodel.qzeros = torch.Tensor([[0]]).to(torch.int32)
    
    inputs = np.random.randn(1, 8).astype("float16")
    inputs = torch.Tensor(inputs).to("cuda")
    print(inputs)
    output = qmodel(inputs)
    print(qmodel.qweight)
    print(qmodel.scales)
    print(qmodel.qzeros)
    print(output)


def mvm_layer(model, name, py=1):
    token = ne.Var("token", 2048)
    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", 2048)

    qweight = model['model.layers.0.self_attn.q_proj.qweight'][:16, :32]
    scales = model['model.layers.0.self_attn.q_proj.scales'][:1, :32]
    input = adr.hbm.var_ddr("input", [1, token, 128])
    weight = adr.hbm.const_hbm("weight", [qweight, scales], [128, 32])
    output = adr.hbm.mvm(input, weight)
    
    glb_attrs = {"kvcache": kvcache, "last_token": last_token}
    output = transform.infer_type(output, dlavm.device.HBM0603, attrs=glb_attrs)
    output = transform.offline(output, os.path.join("output", "gptq_layer"))
    print(output)

    addr_assign = {"runtime": 0x200000000, "hbm": 0x0}
    if py:
        expr, source, storage, _ = codegen.csb_python(output, name, addr_assign)
    else:
        expr, source, storage, _ = codegen.csb_test_head_ops(output, name, addr_assign)
    if py:
        save_path = os.path.join("output", "gptq_layer", name + ".py")
    else:
        save_path = os.path.join("output", "gptq_layer", name + ".h")
    log_path = os.path.join("output", "gptq_layer", name + ".log")
    with open(save_path, "w") as f:
        f.write(source)
    dlavm.utils.LOG_WITH_PREFIX("expression", str(expr))
    dlavm.utils.LOG_WITH_PREFIX("storage", str(storage))
    dlavm.utils.LOG_EXPORT(log_path)
    print(save_path)
    print(log_path)


if __name__ == "__main__":
    model = torch.load("/home/shenao/GPTQ-for-LLaMa/llama2-7b-quant-128.pt")
    mvm_layer(model, "gptq_layer")
    quant_sim(model)