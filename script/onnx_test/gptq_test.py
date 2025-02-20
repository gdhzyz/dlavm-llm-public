import torch
import numpy as np
import quant

np.random.seed(0)


def quant_sim(model):
    qmodel = quant.QuantLinear(bits=4, groupsize=128, infeatures=8, outfeatures=1, bias=False)
    qmodel.qweight = model['model.layers.0.self_attn.q_proj.qweight'][:1, :1]
    # qmodel.qweight = torch.Tensor([[0]]).to(torch.int32)
    qmodel.scales = model['model.layers.0.self_attn.q_proj.scales'][:1, :1]
    # qmodel.scales = torch.Tensor([[0]]).to(torch.float16)
    # qmodel.qzeros = model['model.layers.0.self_attn.q_proj.qzeros'][:1, :1]
    qmodel.qzeros = torch.Tensor([[0]]).to(torch.int32)
    qmodel.g_idx = model['model.layers.0.self_attn.q_proj.g_idx'][:1]
    qmodel = qmodel.to("cuda")
    
    inputs = np.random.randn(1, 8).astype("float16")
    inputs = torch.Tensor(inputs).to("cuda").to(torch.float16)
    print(inputs)
    print(qmodel.qweight)
    output = qmodel(inputs)
    print(qmodel.qweight)
    print(qmodel.scales)
    print(qmodel.qzeros)
    print(output)


if __name__ == "__main__":
    model = torch.load("/home/shenao/GPTQ-for-LLaMa/llama2-7b-quant-128.pt")
    quant_sim(model)