import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

import os
import math
import quant
import random
import numpy as np


def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=True, scale=None) -> torch.Tensor:
    # Efficient implementation equivalent to the following:
    L, S = query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(L, S, dtype=query.dtype).to('cuda')
    if is_causal:
        assert attn_mask is None
        temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0).to('cuda')
        attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
        attn_bias.to(query.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask
    attn_weight = query @ key.transpose(-2, -1)

    scores0 = attn_weight.clone()

    attn_weight = attn_weight * scale_factor

    scores1 = attn_weight.clone()
    Mask = attn_bias.clone()
    rsqrt_dk = scale_factor

    attn_weight += attn_bias

    scores1_5 = attn_weight.clone()

    attn_weight = torch.softmax(attn_weight, dim=-1)
    
    scores2 = attn_weight.clone()

    return attn_weight @ value, scores0, scores1, Mask, rsqrt_dk, scores2, scores1_5

def out_txt_1(data_pt, txt_name):
    data = data_pt.detach().cpu().numpy()
    shape = data.shape

    filename = os.path.join('out_txt', txt_name)
    with open(filename, 'w') as file:
        shape_line = ' '.join(map(str, data.shape))
        file.write(f'{shape_line}\n')
        np.savetxt(file, data.flatten(), delimiter=',', newline='\n')

def out_bin(data_pt, bin_name):
    if data_pt.dtype == torch.float32:
        data_pt = data_pt.to(torch.float16)

    with open(os.path.join('out_txt', bin_name), 'wb') as f:
        f.write(data_pt.detach().cpu().numpy().flatten().tobytes())

def head_chout_chin_2_chout_head_chin(data_pt_src, head, chout, chin):
    data_pt = data_pt_src.reshape((head, chout, chin))
    data_pt = data_pt.permute(1, 0, 2)
    return data_pt

def out_txt(data_pt_src, txt_name, is_out=False, is_transpose=False):
    if is_transpose:
        data_pt = data_pt_src.transpose(0, 1)
    else:
        data_pt = data_pt_src

    data = data_pt.detach().cpu().numpy()
    shape = data.shape
    
    print(txt_name, shape, data.mean(), data_pt.dtype)
    if is_out:
        out_txt_1(data_pt, txt_name)
        out_bin(data_pt, txt_name.replace('.txt', '.bin'))

def quantize_tensor(tensor, num_bits, group_size):
    tensor_grouped = tensor.view(*tensor.shape[:-1], -1, group_size)
    quantized_tensors = []
    scales = []
    for i in range(tensor_grouped.shape[0]):
        group = tensor_grouped[i, ...]
        quantized_group, scale = quantize_group(group, num_bits)
        quantized_tensors.append(quantized_group)
        scales.append(scale)
    quantized_tensors = torch.stack(quantized_tensors, dim=0)
    scales = torch.tensor(scales)
    return quantized_tensors, scales

def quantize_group(tensor, num_bits):
    # 计算最小值和最大值
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    max_val = torch.max(max_val, torch.abs(min_val))

    # 计算量化的范围
    # qmin, qmax = -(2**(num_bits - 1)), (2**(num_bits - 1)) - 1
    # qmin, qmax = 0, 2**num_bits - 1
    scale = 2 * max_val / (2**num_bits - 1)
    # zero_point = qmin - min_val / scale

    # print(max_val / scale, min_val / scale)

    # 量化
    quantized_tensor = (tensor / scale).to(torch.int)
    return quantized_tensor, scale

def dequantize_tensor(quantized_tensor, scales):
    dequantized_tensors = []
    for i in range(quantized_tensor.shape[0]):
        quantized_group = quantized_tensor[i, ...]
        scale = scales[i]
        dequantized_group = dequantize_group(quantized_group, scale)
        dequantized_tensors.append(dequantized_group)
    dequantized_tensors = torch.stack(dequantized_tensors, dim=0)
    return dequantized_tensors.view(*dequantized_tensors.shape[:-2], -1)

def dequantize_group(quantized_tensor, scale):
    # 计算量化的范围
    # qmin, qmax = -(2**(num_bits - 1)), (2**(num_bits - 1)) - 1
    # 反量化
    dequantized_tensor = quantized_tensor * scale
    return dequantized_tensor

def quant_minmax(weight, _bits, _gs):
    # weight = torch.rand(4096, 4096)
    h, w = weight.shape
    qweight = torch.zeros(h, w)
    qscales = torch.zeros(h//_gs, w)
    
    for i in range(0, h, _gs):
        for j in range(w):
            block = weight[i:i+_gs, j]
            qblock, scale = quantize_group(block, _bits)
            qweight[i:i+_gs, j] = qblock
            qscales[i//_gs, j] = scale

    return qweight, qscales

def split_tensor2(int32_value, dim=0):
    mask = torch.tensor(0xF, dtype=torch.int32)
    if dim == 0:
        int4_values = []
        for n in int32_value:
            m = n.view(-1, 1) >> (torch.arange(0, 32, 4)) & mask
            int4_values.append(m.t())
        int4_values = torch.cat(int4_values, dim=0)
        return int4_values.int()
    elif dim == 1:
        int4_values = []
        for n in int32_value.t():
            m = n.view(-1, 1) >> (torch.arange(0, 32, 4)) & mask
            int4_values.append(m.t())
        int4_values = torch.cat(int4_values, dim=0)
        return int4_values.t().int()

import pickle
from gptq import GPTQ, Observer

def in_pickle(pkl_name):
    print(pkl_name)
    weight = None
    with open(pkl_name, 'rb') as f:
        weight = pickle.load(f)
    return weight

def quant_layer(_layer, _wbits, _groupsize):
    _layer.to('cuda')
    inps = torch.randn(19, 1, _layer.in_features).to(torch.float16).to('cuda')
    outs = torch.zeros_like(inps)
    output1 = _layer(inps)

    gptq = GPTQ(_layer, observe=True)
    gptq.add_batch(inps, outs)
    gptq.quantizer.configure(
        _wbits, perchannel=True, sym=True, mse=False)
    scale, zero, g_idx, trace = gptq.fasterquant(
        percdamp=0.1, groupsize=_groupsize, actorder=False)

    gptq.layer.to('cpu')
    gptq.free()

    _layer.to('cpu')
    _in = _layer.in_features
    _out = _layer.out_features
    _bias = _layer.bias
    
    qlinear = quant.QuantLinear(
        bits=_wbits, groupsize=_groupsize, infeatures=_in, outfeatures=_out, bias=_bias is not None)
    qlinear.pack(_layer, scale.cpu(), zero.cpu(), g_idx.cpu())
    
    scale, zero, g_idx, error = None, None, None, None

    qlinear.to('cuda')
    output2 = qlinear(inps)
    output2 = output2

    similarity = F.cosine_similarity(output1.to(torch.float32), output2.to(torch.float32), dim=-1)

    return qlinear, torch.mean(similarity).item()

def gpqt_layer_qkv(idx):
    weight = in_pickle(f'pkl_glm2/self_attention.query_key_value_{idx}.pkl')
    bias = in_pickle(f'pkl_glm2/self_attention.query_key_value_bias_{idx}.pkl')
    qkv = nn.Linear(in_features=4096, out_features=4608, bias=True, dtype=torch.float16)
    qkv.weight = torch.nn.Parameter(torch.tensor(weight).to(torch.float16))
    qkv.bias = torch.nn.Parameter(torch.tensor(bias).to(torch.float16))   
    layer, value = quant_layer(qkv, 4, 128)
    return layer

def gpqt_layer_dense(idx):
    weight = in_pickle(f'pkl_glm2/self_attention.dense_{idx}.pkl')
    dense = nn.Linear(in_features=4096, out_features=4096, bias=False, dtype=torch.float16)
    dense.weight = torch.nn.Parameter(torch.tensor(weight).to(torch.float16))
    layer, value = quant_layer(dense, 4, 128)
    return layer

def gpqt_layer_dense_h_4h(idx):
    weight = in_pickle(f'pkl_glm2/mlp.dense_h_to_4h_{idx}.pkl')
    dense = nn.Linear(in_features=4096, out_features=27392, bias=False, dtype=torch.float16)
    dense.weight = torch.nn.Parameter(torch.tensor(weight).to(torch.float16))
    layer, value = quant_layer(dense, 4, 128)
    return layer

def gpqt_layer_dense_4h_h(idx):
    weight = in_pickle(f'pkl_glm2/mlp.dense_4h_to_h_{idx}.pkl')
    dense = nn.Linear(in_features=13696, out_features=4096, bias=False, dtype=torch.float16)
    dense.weight = torch.nn.Parameter(torch.tensor(weight).to(torch.float16))
    layer, value = quant_layer(dense, 4, 128)
    return layer 

@torch.jit.script
def apply_rotary_pos_emb(x: torch.Tensor, rope_cache: torch.Tensor) -> torch.Tensor:
    # x: [sq, b, np, hn]
    sq, b, np, hn = x.size(0), x.size(1), x.size(2), x.size(3)
    rot_dim = rope_cache.shape[-2] * 2
    x, x_pass = x[..., :rot_dim], x[..., rot_dim:]
    # truncate to support variable sizes
    rope_cache = rope_cache[:sq]
    xshaped = x.reshape(sq, -1, np, rot_dim // 2, 2)
    rope_cache = rope_cache.view(sq, -1, 1, xshaped.size(3), 2)

    x_out2 = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0] -
            xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0] +
            xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    output = torch.cat((x_out2, x_pass), dim=-1)
    return output


class RMSNorm(torch.nn.Module): 
    def __init__(self, normalized_shape, eps=1e-5, device=None, dtype=None, **kwargs):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.empty(
            normalized_shape, device=device, dtype=dtype))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(
            2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        return (self.weight * hidden_states).to(input_dtype)

class QuantMinMaxLinear(torch.nn.Module):
    def __init__(self, qweight, qscales, bias=0):
        super().__init__()
        self.qweight = qweight
        self.qscales = qscales
        self.bias = bias

    def forward(self, hidden_states: torch.Tensor):
        scales = self.qscales.repeat_interleave(128, dim=0)
        data_in = (self.qweight * scales).to(torch.float16).cuda()
        output = hidden_states.matmul(data_in) + self.bias
        return output

class MyGML2(nn.Module):
    def __init__(self, device='cuda'):
        super(MyGML2, self).__init__()
        self.device = device
        self.seq_length = 32768
        self.word_embeddings = nn.Embedding(
            65024, 4096, dtype=torch.float16, device=device)

        self.dim = 64
        self.input_layernorm = torch.nn.ModuleList()
        self.post_attention_layernorm = torch.nn.ModuleList()
        self.query_key_value = torch.nn.ModuleList()
        self.dense = torch.nn.ModuleList()
        self.dense_h_to_4h = torch.nn.ModuleList()
        self.dense_4h_to_h = torch.nn.ModuleList()

        for i in range(28):
            self.input_layernorm.append(
                RMSNorm(4096, device=device, dtype=torch.float16))
            self.post_attention_layernorm.append(
                RMSNorm(4096, device=device, dtype=torch.float16))

            self.query_key_value.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=4096, outfeatures=4608, bias=True).to(device))
            self.dense.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=4096, outfeatures=4096, bias=False).to(device))

            self.dense_h_to_4h.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=4096, outfeatures=27392, bias=False).to(device))
            self.dense_4h_to_h.append(quant.QuantLinear(
                bits=4, groupsize=128, infeatures=13696, outfeatures=4096, bias=False).to(device))

        self.final_layernorm = RMSNorm(
            4096, device=device, dtype=torch.float16)
        self.output_layer = nn.Linear(
            4096, 65024, bias=False, dtype=torch.float16)

    def swiglu(self, x):
        x = torch.chunk(x, 2, dim=-1)
        return F.silu(x[0]) * x[1]

    def forward(self, inputs):
        embeddings = self.word_embeddings(inputs["input_ids"])
        out_txt(embeddings, 'embeddings.txt')

        embeddings = embeddings.transpose(0, 1).contiguous()
        embeddings.to(torch.float16)
        # out_txt(embeddings, 'embeddings_T.txt')
        out_txt(embeddings, 'chatglm_page1/Embedded_input.txt', is_out=True)

        theta = 1.0 / (10000 ** (torch.arange(0, self.dim, 2,
                                              dtype=torch.float16, device=self.device) / self.dim))
        out_txt(theta, 'theta.txt')

        seq_idx = torch.arange(
            self.seq_length, dtype=torch.float16, device=self.device)
        out_txt(seq_idx, 'seq_idx.txt')

        idx_theta = torch.outer(seq_idx, theta).float()
        out_txt(idx_theta, 'idx_theta.txt')

        rotary_pos_emb = torch.stack(
            [torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1).half()
        out_txt(rotary_pos_emb, 'rotary_pos_emb.txt')

        rotary_pos_emb = rotary_pos_emb[inputs["position_ids"]]
        out_txt(rotary_pos_emb, 'rotary_pos_emb_in.txt')
        # out_txt(rotary_pos_emb.reshape(128, 64), 'pos_in.txt', is_out=True)

        rotary_pos_emb = rotary_pos_emb.transpose(0, 1).contiguous()
        out_txt(rotary_pos_emb, 'rotary_pos_emb_in_T.txt')

        hidden_states = embeddings

        ratio = 10.0

        for i in range(28):
            torch.cuda.empty_cache()

            blockname = 'BLOCK' + str(i).zfill(2) + '/'
            # page1
            if i==0:
                self.input_layernorm[i].weight.data = self.input_layernorm[i].weight.data * ratio

            layernorm_output = self.input_layernorm[i](hidden_states)

            # print(i, self.input_layernorm[i].weight.data.mean(), hidden_states.mean(), layernorm_output.mean())
            # out_txt(self.input_layernorm[i].weight, "input_layernorm_{}.txt".format(i))
            # out_txt(layernorm_output, "layernorm_output_{}.txt".format(i))

            out_txt(self.input_layernorm[i].weight, blockname + f"chatglm_page1/LN_k.txt", is_out=True)
            out_txt(layernorm_output, blockname + f"chatglm_page1/Layer_Norm_output.txt", is_out=True)

            if i==0:
                self.input_layernorm[i].weight.data = self.input_layernorm[i].weight.data / ratio
            # exit()

            # self.query_key_value[i] = gpqt_layer_qkv(i)
            # qkv = split_tensor2(self.query_key_value[i].qweight.cpu())
            # qkv = torch.clamp(qkv - 7, -7, 7)
            # scales = self.query_key_value[i].scales
            # bias = self.query_key_value[i].bias
            # print(qkv.shape, scales.shape, bias.shape)

            
            qkv, scales, bias = self.query_key_value[i].qweight, self.query_key_value[i].qscales, self.query_key_value[i].bias

            Wqkv = qkv.split([128 * 32, 128 * 2, 128 * 2], dim=-1, )
            Wq = head_chout_chin_2_chout_head_chin(Wqkv[0].transpose(0, 1), 32, 128, 4096)
            Wk = head_chout_chin_2_chout_head_chin(Wqkv[1].transpose(0, 1), 2, 128, 4096)
            Wv = head_chout_chin_2_chout_head_chin(Wqkv[2].transpose(0, 1), 2, 128, 4096)
            out_txt(Wq, blockname + f"chatglm_page1/MVM_BN_Wq.txt", is_out=True)
            out_txt(Wk, blockname + f"chatglm_page1/MVM_BN_Wk.txt", is_out=True)
            out_txt(Wv, blockname + f"chatglm_page1/MVM_BN_Wv.txt", is_out=True)

            Scale = scales.split([128 * 32, 128 * 2, 128 * 2], dim=-1, )
            Scaleq = head_chout_chin_2_chout_head_chin(Scale[0].transpose(0, 1), 32, 128, 32)
            Scalek = head_chout_chin_2_chout_head_chin(Scale[1].transpose(0, 1), 2, 128, 32)
            Scalev = head_chout_chin_2_chout_head_chin(Scale[2].transpose(0, 1), 2, 128, 32)
            out_txt(Scaleq, blockname + f"chatglm_page1/MVM_BN_Scaleq.txt", is_out=True)
            out_txt(Scalek, blockname + f"chatglm_page1/MVM_BN_Scalek.txt", is_out=True)
            out_txt(Scalev, blockname + f"chatglm_page1/MVM_BN_Scalev.txt", is_out=True)

            Bias = bias.split([128 * 32, 128 * 2, 128 * 2], dim=-1, )
            out_txt(Bias[0], blockname + f"chatglm_page1/MVM_BN_Biasq.txt", is_out=True)
            out_txt(Bias[1], blockname + f"chatglm_page1/MVM_BN_Biask.txt", is_out=True)
            out_txt(Bias[2], blockname + f"chatglm_page1/MVM_BN_Biasv.txt", is_out=True)

            Wqkv = qkv.split([128] * 36, dim=-1, )
            for ii, w in enumerate(Wqkv):
                if ii < 32:
                    out_txt(w, blockname + f"chatglm_page1/MVM_BN_Wq_{ii}.txt", is_out=True)
                elif ii < 34:
                    out_txt(w, blockname + f"chatglm_page1/MVM_BN_Wk_{ii-32}.txt", is_out=True)
                else:
                    out_txt(w, blockname + f"chatglm_page1/MVM_BN_Wv_{ii-34}.txt", is_out=True)

            Scale = scales.split([128] * 36, dim=-1, )
            for ii, sl in enumerate(Scale):
                if ii < 32:
                    out_txt(sl, blockname + f"chatglm_page1/MVM_BN_Scaleq_{ii}.txt", is_out=True)
                elif ii < 34:
                    out_txt(sl, blockname + f"chatglm_page1/MVM_BN_Scalek_{ii-32}.txt", is_out=True)
                else:
                    out_txt(sl, blockname + f"chatglm_page1/MVM_BN_Scalev_{ii-34}.txt", is_out=True)

            Bias = bias.split([128] * 36, dim=-1, )
            for ii, b in enumerate(Bias):
                if ii < 32:
                    out_txt(b, blockname + f"chatglm_page1/MVM_BN_Biasq_{ii}.txt", is_out=True)
                elif ii < 34:
                    out_txt(b, blockname + f"chatglm_page1/MVM_BN_Biask_{ii-32}.txt", is_out=True)
                else:
                    out_txt(b, blockname + f"chatglm_page1/MVM_BN_Biasv_{ii-34}.txt", is_out=True)

            mixed_x_layer = self.query_key_value[i](layernorm_output)
            # out_txt(self.query_key_value[i].qweight, "query_key_value_qweight_{}.txt".format(i))
            # out_txt(self.query_key_value[i].qzeros, "query_key_value_qzeros_{}.txt".format(i))
            # out_txt(self.query_key_value[i].scales, "query_key_value_scales_{}.txt".format(i))
            # out_txt(self.query_key_value[i].g_idx, "query_key_value_g_idx_{}.txt".format(i))
            # out_txt(self.query_key_value[i].bias, "query_key_value_bias_{}.txt".format(i))
            out_txt(mixed_x_layer, "mixed_x_layer_{}.txt".format(i))

            (query_layer, key_layer, value_layer) = mixed_x_layer.split(
                [32 * 128, 2 * 128, 2 * 128, ], dim=-1, )
            out_txt(query_layer, "query_layer_{}.txt".format(i))
            out_txt(key_layer, "key_layer_{}.txt".format(i))
            out_txt(value_layer, "value_layer_{}.txt".format(i))

            query_layer = query_layer.view(query_layer.size()[:-1] + (32, 128))
            key_layer = key_layer.view(key_layer.size()[:-1] + (2, 128))
            value_layer = value_layer.view(value_layer.size()[:-1] + (2, 128))

            Query = query_layer.reshape(-1, 32, 128).permute(0, 2, 1)
            Key = key_layer.reshape(-1, 2, 128).permute(0, 2, 1)
            Value = value_layer.reshape(-1, 2, 128).permute(0, 2, 1)

            out_txt(Query, blockname + f"chatglm_page1/Query.txt", is_out=True)
            out_txt(Key, blockname + f"chatglm_page1/Key.txt", is_out=True)
            out_txt(Value, blockname + f"chatglm_page1/Value.txt", is_out=True)

            for ii in range(32):
                Query = query_layer[:,:,ii,:]
                out_txt(Query, blockname + f"chatglm_page1/Query_{ii}.txt", is_out=True)
                if ii < 2:
                    Key = key_layer[:,:,ii,:]
                    out_txt(Key, blockname + f"chatglm_page1/Key_{ii}.txt", is_out=True)
                    
                    Value = value_layer[:,:,ii,:]
                    out_txt(Value, blockname + f"chatglm_page1/Value_{ii}.txt", is_out=True)
            

            query_layer = apply_rotary_pos_emb(query_layer, rotary_pos_emb)
            key_layer = apply_rotary_pos_emb(key_layer, rotary_pos_emb)
            out_txt(query_layer, "query_layer_view_emb_{}.txt".format(i))
            out_txt(key_layer, "key_layer_view_emb_{}.txt".format(i))

            out_txt(query_layer, blockname + f"chatglm_page1/Emb_Query.txt", is_out=True)
            out_txt(key_layer, blockname + f"chatglm_page1/Emb_Key.txt", is_out=True)

            
            for ii in range(32):
                Emb_Query = query_layer[:,:,ii,:]
                out_txt(Emb_Query, blockname + f"chatglm_page1/Emb_Query_{ii}.txt", is_out=True)
                if ii < 2:
                    Emb_Key = key_layer[:,:,ii,:]
                    out_txt(Emb_Key, blockname + f"chatglm_page1/Emb_Key_{ii}.txt", is_out=True)
            
            # exit()

            # page2
            for ii in range(2):
                Emb_Key = key_layer[:,:,ii,:]
                Value = value_layer[:,:,ii,:]

                # Emb_Key, scales_k = quantize_tensor(Emb_Key, num_bits=4, group_size=128)
                # Value, scales_v = quantize_tensor(Value, num_bits=4, group_size=128)
                # Emb_Key = dequantize_tensor(Emb_Key, scales_k)
                # Value = dequantize_tensor(Value, scales_v)
                
                # out_txt(Emb_Key, blockname + f"chatglm_page2/Quan_Key_{ii}.txt", is_out=True)
                # out_txt(scales_k, blockname + f"chatglm_page2/Quan_Key_scales_{ii}.txt", is_out=True)
                # out_txt(Value, blockname + f"chatglm_page2/Quan_Value_{ii}.txt", is_out=True)

            # key_layer, scales_k = quantize_tensor(key_layer, num_bits=4, group_size=128)
            # value_layer, scales_v = quantize_tensor(value_layer, num_bits=4, group_size=128)
            # key_layer = dequantize_tensor(key_layer, scales_k)
            # value_layer = dequantize_tensor(value_layer, scales_v)

            key_layer = key_layer.unsqueeze(-2)
            key_layer = key_layer.expand(-1, -1, -1, 32 // 2, -1)
            key_layer = key_layer.contiguous().view(
                key_layer.size()[:2] + (32, 128))
            out_txt(key_layer, "key_layer_view_emb_uec{}.txt".format(i))
            
            value_layer = value_layer.unsqueeze(-2)
            value_layer = value_layer.expand(-1, -1, -1, 32 // 2, -1)
            value_layer = value_layer.contiguous().view(
                value_layer.size()[:2] + (32, 128))
            out_txt(value_layer, "value_layer_view_emb_uec{}.txt".format(i))
            
            query_layer, key_layer, value_layer = [
                k.permute(1, 2, 0, 3) for k in [query_layer, key_layer, value_layer]]

            if i==0:
                query_layer, key_layer, value_layer = query_layer / ratio, key_layer / ratio, value_layer / ratio

            context_layer = torch.nn.functional.scaled_dot_product_attention(
                query_layer, key_layer, value_layer, is_causal=True
            out_txt(context_layer, blockname + f"chatglm_page2/Attention_output.txt", is_out=True)

            '''
            context_layer2, scores0, scores1, Mask, rsqrt_dk, scores2, scores1_5 = scaled_dot_product_attention(query_layer, key_layer, value_layer)
            out_txt(scores0, blockname + f"chatglm_page2/scores0.txt", is_out=True)
            out_txt(scores1_5, blockname + f"chatglm_page2/scores1_5.txt", is_out=True)
            out_txt(scores1_5[0][0], blockname + f"chatglm_page2/scores1_5_index0.txt", is_out=True)
            out_txt(scores1, blockname + f"chatglm_page2/scores1.txt", is_out=True)
            Mask[Mask == float('-inf')] = torch.tensor(np.float16(-65504), dtype=torch.float16)
            Mask = torch.stack([Mask] * 32, dim=0)
            out_txt(Mask, blockname + f"chatglm_page2/Mask.txt", is_out=True)
            print('rsqrt_dk', rsqrt_dk)
            out_txt(scores2, blockname + f"chatglm_page2/scores2.txt", is_out=True)
            out_txt(context_layer2, blockname + f"chatglm_page2/Attention_output.txt", is_out=True)

            out_txt(context_layer, "context_layer_{}.txt".format(i))
            for ii in range(32):
                Attention_output = context_layer[0][ii]
                out_txt(Attention_output, blockname + f"chatglm_page2/Attention_output_{ii}.txt", is_out=True)
            '''

            # page3
            context_layer = context_layer.permute(2, 0, 1, 3)
            new_context_layer_shape = context_layer.size()[:-2] + (128 * 32, )
            context_layer = context_layer.reshape(*new_context_layer_shape)
            out_txt(context_layer, "context_layer_reshape_{}.txt".format(i))

            attention_output = self.dense[i](context_layer)
            out_txt(attention_output, "attention_output_{}.txt".format(i))

            residual = hidden_states
            layernorm_input = torch.nn.functional.dropout(
                attention_output, p=0.0, training=False)
            layernorm_input = residual + layernorm_input
            out_txt(layernorm_input, "layernorm_input_{}.txt".format(i))

            out_txt(self.dense[i].qweight, blockname + f"chatglm_page3/MVM_BN_RES_weight.txt", is_out=True, is_transpose=True)
            out_txt(self.dense[i].qscales, blockname + f"chatglm_page3/MVM_BN_RES_scales.txt", is_out=True, is_transpose=True)
            out_txt(layernorm_input, blockname + f"chatglm_page3/MVM_BN_RES_output.txt", is_out=True)

            # page4
            layernorm_output_att = self.post_attention_layernorm[i](
                layernorm_input)
            out_txt(layernorm_output_att, "layernorm_output_att_{}.txt".format(i))
            out_txt(self.post_attention_layernorm[i].weight, blockname + f"chatglm_page4/post_attention_layernorm_weight.txt", is_out=True)
            out_txt(layernorm_output_att, blockname + f"chatglm_page4/post_norm_output.txt", is_out=True)
            
            h_4h_weight = self.dense_h_to_4h[i].qweight.split([13696, 13696], dim=-1)
            out_txt(h_4h_weight[0], blockname + f"chatglm_page4/dense_h_to_4h_weight_0.txt", is_out=True, is_transpose=True)
            out_txt(h_4h_weight[1], blockname + f"chatglm_page4/dense_h_to_4h_weight_1.txt", is_out=True, is_transpose=True)

            h_4h_scale = self.dense_h_to_4h[i].qscales.split([13696, 13696], dim=-1)
            out_txt(h_4h_scale[0], blockname + f"chatglm_page4/dense_h_to_4h_scales_0.txt", is_out=True, is_transpose=True)
            out_txt(h_4h_scale[1], blockname + f"chatglm_page4/dense_h_to_4h_scales_1.txt", is_out=True, is_transpose=True)

            # out_txt(self.dense_h_to_4h[i].qweight, blockname + f"chatglm_page4/dense_h_to_4h_weight.txt", is_out=True, is_transpose=True)
            # out_txt(self.dense_h_to_4h[i].qscales, blockname + f"chatglm_page4/dense_h_to_4h_scales.txt", is_out=True, is_transpose=True)

            intermediate_parallel = self.dense_h_to_4h[i](layernorm_output_att)
            out_txt(intermediate_parallel, "intermediate_parallel_{}.txt".format(i))
            out_txt(intermediate_parallel, blockname + f"chatglm_page4/Dense_4h_output.txt", is_out=True)

            # intermediate_parallel = self.swiglu(intermediate_parallel)
            imps = torch.chunk(intermediate_parallel, 2, dim=-1)
            Ele_output = F.silu(imps[0])
            intermediate_parallel = Ele_output * imps[1]
            out_txt(Ele_output, blockname + f"chatglm_page4/Ele_output.txt", is_out=True)
            out_txt(imps[0], blockname + f"chatglm_page4/chunk_output0.txt", is_out=True)
            out_txt(imps[1], blockname + f"chatglm_page4/chunk_output1.txt", is_out=True)

            out_txt(intermediate_parallel, "intermediate_parallel_swiglu_{}.txt".format(i))
            out_txt(intermediate_parallel, blockname + f"chatglm_page4/ACT_output.txt", is_out=True)
            
            out_txt(self.dense_4h_to_h[i].qweight, blockname + f"chatglm_page4/dense_4h_to_h_weight.txt", is_out=True, is_transpose=True)
            out_txt(self.dense_4h_to_h[i].qscales, blockname + f"chatglm_page4/dense_4h_to_h_scales.txt", is_out=True, is_transpose=True)

            mlp_output = self.dense_4h_to_h[i](intermediate_parallel)
            out_txt(mlp_output, "mlp_output_{}.txt".format(i))
            out_txt(mlp_output, blockname + f"chatglm_page4/Dense_h_output.txt", is_out=True)

            residual = layernorm_input
            hidden_states = torch.nn.functional.dropout(
                mlp_output, p=0.0, training=False)
            hidden_states = residual + hidden_states
            out_txt(hidden_states, "hidden_states_{}.txt".format(i))
            exit()

        # exit()
        hidden_states = self.final_layernorm(hidden_states)
        out_txt(hidden_states, "hidden_states_end.txt")

        # hidden_states = hidden_states[-1:]
        lm_logits = self.output_layer(hidden_states)
        out_txt(lm_logits, "lm_logits.txt")

        lm_logits = lm_logits.transpose(0, 1).contiguous()

        return lm_logits

def layer2quant(layer):
    qweight, qscales = quant_minmax(layer.weight.transpose(0, 1), 4, 128)
    bias = layer.bias if layer.bias is not None else 0
    qmml = QuantMinMaxLinear(qweight, qscales, bias)
    return qmml

def load_state_dict(model):
    for i in range(0, 28, 1):
        idx = str(i)
        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.self_attention.query_key_value.pkl'
        model.query_key_value[i] = in_pickle(pklname.replace('0', idx))
        
        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.self_attention.dense.pkl'
        model.dense[i] = in_pickle(pklname.replace('0', idx))

        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.mlp.dense_h_to_4h.pkl'
        model.dense_h_to_4h[i] = in_pickle(pklname.replace('0', idx))

        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.mlp.dense_4h_to_h.pkl'
        model.dense_4h_to_h[i] = in_pickle(pklname.replace('0', idx))

        # quant
        '''
        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.self_attention.query_key_value.pkl'
        model.query_key_value[i] = layer2quant(model.query_key_value[i])
        with open(pklname.replace('0', idx), 'wb') as f:
            pickle.dump(model.query_key_value[i], f)

        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.self_attention.dense.pkl'
        model.dense[i] = layer2quant(model.dense[i])
        with open(pklname.replace('0', idx), 'wb') as f:
            pickle.dump(model.dense[i], f)

        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.mlp.dense_h_to_4h.pkl'
        model.dense_h_to_4h[i] = layer2quant(model.dense_h_to_4h[i])
        with open(pklname.replace('0', idx), 'wb') as f:
            pickle.dump(model.dense_h_to_4h[i], f)

        pklname = 'layers_quant_minmax/transformer.encoder.layers.0.mlp.dense_4h_to_h.pkl'   
        model.dense_4h_to_h[i] = layer2quant(model.dense_4h_to_h[i])
        with open(pklname.replace('0', idx), 'wb') as f:
            pickle.dump(model.dense_4h_to_h[i], f)
        '''

    pklname = 'layers_source/transformer.output_layer.pkl'
    model.output_layer = in_pickle(pklname)

    # torch.cuda.empty_cache()

def main():
    device = 'cuda'
    query = '深圳有哪些景点'
    tokenizer = AutoTokenizer.from_pretrained(
        "mytokens", trust_remote_code=True, local_files_only=True)

    model = MyGML2(device=device)
    print('model create complete.')

    model.load_state_dict(torch.load('MyGLM2-4bit-128g.pt'), strict=True)
    load_state_dict(model)
    print('model load weight complete.')

    model.to(device)
    model.eval()

    '''
    out_txt(model.final_layernorm.weight, f"final_layernorm_LN_k.txt", is_out=True)
    output_layer = split_tensor2(model.output_layer.qweight.cpu())
    output_layer = torch.clamp(output_layer - 7, -7, 7)
    scales = model.output_layer.scales
    out_txt(output_layer, f"output_layer_weights.txt", is_out=True)
    out_txt(scales, f"output_layer_scales.txt", is_out=True)
    exit()
    '''

    print('model infer...')
    generated_text = ''
    for i in range(500):
        prompt = "[Round {}]\n\n问：{}\n\n答：".format(
            1, query) + generated_text
        # [Round 1]\n\n问：深圳有哪些景点\n\n答：
        inputs = tokenizer([prompt], return_tensors="pt")

        out_txt(inputs['input_ids'], 'input_ids.txt')
        out_txt(inputs['attention_mask'], 'attention_mask.txt')
        out_txt(inputs['position_ids'], 'position_ids.txt')

        inputs = inputs.to(device)
        outputs = model(inputs)[0][-1:]

        next_tokens = torch.argmax(outputs, dim=-1)
        # _, next_tokens = torch.topk(outputs, k=5, dim=-1)
        # next_tokens = next_tokens[0, 0]
        if next_tokens == 2:
            break
        next_tokens = tokenizer.decode(next_tokens)
        next_tokens = ' ' if len(next_tokens) == 0 else next_tokens
        generated_text += next_tokens
        print(next_tokens, end='')
        # torch.cuda.empty_cache()
        # break

    print()
    print(prompt)

def get_txt(txtname, pass_first=True):
    datas = []
    with open(txtname, 'r', encoding='utf8') as f:
        lines = f.readlines()
        if pass_first:
            del lines[0]
        for line in lines:
            if len(line.strip('\n')) >= 1:
                value = float(line)
                datas.append(value)
    return datas

def check_error(data1, data2, relative_threshold=0.01, absolute_threshold=0.01):
    """
    Check if the average relative error is less than a percentage threshold and average absolute error
    is less than a given threshold.

    Parameters:
    - data1: First set of data
    - data2: Second set of data
    - relative_threshold: Percentage threshold for average relative error (default is 0.01, i.e., 1%)
    - absolute_threshold: Absolute threshold for average absolute error (default is 0.01)

    Returns:
    - True if both average relative and absolute errors meet the criteria, False otherwise
    """
    absolute_error = torch.abs(data1 - data2)
    relative_error = absolute_error / (torch.abs(data1) + torch.abs(data2) + 1e-12)  # Avoid division by zero

    average_relative_error = torch.mean(relative_error)
    average_absolute_error = torch.mean(absolute_error)

    return '相对误差:' + str(average_relative_error) + '绝对误差:' + str(average_absolute_error)

def weight_resort_QKV():
    pklname = 'layers_source/transformer.encoder.layers.0.self_attention.query_key_value.pkl'
    query_key_value = in_pickle(pklname).cuda()
    Wt = query_key_value.weight
    bias = query_key_value.bias

    seq_len = 19
    layernorm_output = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Layer_Norm_output.txt')
    layernorm_output = torch.tensor(layernorm_output).reshape(seq_len, 4096).to(torch.float16).cuda()

    Query = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Query.txt')
    Query = torch.tensor(Query).reshape(seq_len, 128, 32).to(torch.float16).cuda()
    Key = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Key.txt')
    Key = torch.tensor(Key).reshape(seq_len, 128, 2).to(torch.float16).cuda()
    Value = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Value.txt')
    Value = torch.tensor(Value).reshape(seq_len, 128, 2).to(torch.float16).cuda()
    Query = torch.cat((Query, Key, Value), dim=2)

    # Wt = Wt.reshape(2, 128, 4096).permute(1, 0, 2).reshape(256, 4096)
    Wt = Wt.reshape(36, 128, 4096).permute(1, 0, 2).reshape(4608, 4096)
    Wt, scales = quant_minmax(Wt.transpose(0, 1), 4, 128)

    out_txt(Wt.to(torch.int32), 'BLOCK00/chatglm_page1/MVM_BN_Wqkv.txt', is_out=True, is_transpose=True)
    out_txt(scales, 'BLOCK00/chatglm_page1/MVM_BN_Scaleqkv.txt', is_out=True, is_transpose=True)
    out_txt(bias.reshape(36, 128).permute(1, 0), 'BLOCK00/chatglm_page1/MVM_BN_Biasqkv.txt', is_out=True)

    # Wt = Wt.reshape(4096, 128, 2).permute(0, 2, 1).reshape(4096, 256)
    # scales = scales.reshape(32, 128, 2).permute(0, 2, 1).reshape(32, 256)
    Wt = Wt.reshape(4096, 128, 36).permute(0, 2, 1).reshape(4096, 4608)
    scales = scales.reshape(32, 128, 36).permute(0, 2, 1).reshape(32, 4608)

    qmml = QuantMinMaxLinear(Wt, scales, bias)
    Query2 = qmml(layernorm_output).reshape(seq_len, 36, 128).permute(0, 2, 1)

    out_txt(Query.to(torch.float16), 'BLOCK00/chatglm_page1/Query_Key_Value.txt', is_out=True)
    print(Query)
    print(Query2)
    print(Query.shape, Query2.shape) 

def weight_resort_Q():
    pklname = 'layers_source/transformer.encoder.layers.0.self_attention.query_key_value.pkl'
    query_key_value = in_pickle(pklname).cuda()
    Wt = query_key_value.weight.split([128 * 32, 128 * 2, 128 * 2], dim=0, )[0]
    bias = query_key_value.bias.split([128 * 32, 128 * 2, 128 * 2], dim=0, )[0]

    seq_len = 19
    layernorm_output = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Layer_Norm_output.txt')
    layernorm_output = torch.tensor(layernorm_output).reshape(seq_len, 4096).to(torch.float16).cuda()

    Query = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Query.txt')
    Query = torch.tensor(Query).reshape(seq_len, 128, 32).to(torch.float16).cuda()

    # Wt = Wt.reshape(2, 128, 4096).permute(1, 0, 2).reshape(256, 4096)
    Wt = Wt.reshape(32, 128, 4096).permute(1, 0, 2).reshape(4096, 4096)
    Wt, scales = quant_minmax(Wt.transpose(0, 1), 4, 128)

    out_txt(Wt.to(torch.int32), 'BLOCK00/chatglm_page1/MVM_BN_Wq.txt', is_out=True, is_transpose=True)
    out_txt(scales, 'BLOCK00/chatglm_page1/MVM_BN_Scaleq.txt', is_out=True, is_transpose=True)
    out_txt(bias.reshape(32, 128).permute(1, 0), 'BLOCK00/chatglm_page1/MVM_BN_Biasq.txt', is_out=True)

    # Wt = Wt.reshape(4096, 128, 2).permute(0, 2, 1).reshape(4096, 256)
    # scales = scales.reshape(32, 128, 2).permute(0, 2, 1).reshape(32, 256)
    Wt = Wt.reshape(4096, 128, 32).permute(0, 2, 1).reshape(4096, 4096)
    scales = scales.reshape(32, 128, 32).permute(0, 2, 1).reshape(32, 4096)

    qmml = QuantMinMaxLinear(Wt, scales, bias)
    Query2 = qmml(layernorm_output).reshape(seq_len, 32, 128).permute(0, 2, 1)

    print(Query)
    print(Query2)
    print(Query.shape, Query2.shape)    

def read_bin(bin_name):
    # 从文件中读取数据
    with open(bin_name, 'rb') as f:
        data_bytes = f.read()
    
    # 将字节数据转换为NumPy数组
    data_np = np.frombuffer(data_bytes, dtype=np.float16)
    return data_np

def weight_resort_K():
    pklname = 'layers_source/transformer.encoder.layers.0.self_attention.query_key_value.pkl'
    query_key_value = in_pickle(pklname).cuda()
    Wt = query_key_value.weight.split([128 * 32, 128 * 2, 128 * 2], dim=0, )[1]
    bias = query_key_value.bias.split([128 * 32, 128 * 2, 128 * 2], dim=0, )[1]

    seq_len = 19
    layernorm_output = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Layer_Norm_output.txt')
    layernorm_output = torch.tensor(layernorm_output).reshape(seq_len, 4096).to(torch.float16).cuda()

    Key = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Key.txt')
    Key = torch.tensor(Key).reshape(seq_len, 128, 2).to(torch.float16).cuda()

    Wt = Wt.reshape(2, 128, 4096).permute(1, 0, 2).reshape(256, 4096)
    # Wt = Wt.reshape(32, 128, 4096).permute(1, 0, 2).reshape(4096, 4096)
    Wt, scales = quant_minmax(Wt.transpose(0, 1), 4, 128)

    out_txt(Wt.to(torch.int32), 'BLOCK00/chatglm_page1/MVM_BN_Wk.txt', is_out=True, is_transpose=True)
    out_txt(scales, 'BLOCK00/chatglm_page1/MVM_BN_Scalek.txt', is_out=True, is_transpose=True)
    out_txt(bias.reshape(2, 128).permute(1, 0), 'BLOCK00/chatglm_page1/MVM_BN_Biask.txt', is_out=True)

    Wt = Wt.reshape(4096, 128, 2).permute(0, 2, 1).reshape(4096, 256)
    scales = scales.reshape(32, 128, 2).permute(0, 2, 1).reshape(32, 256)
    # Wt = Wt.reshape(4096, 128, 32).permute(0, 2, 1).reshape(4096, 4096)
    # scales = scales.reshape(32, 128, 32).permute(0, 2, 1).reshape(32, 4096)

    qmml = QuantMinMaxLinear(Wt, scales, bias)
    Key2 = qmml(layernorm_output).reshape(seq_len, 2, 128).permute(0, 2, 1)

    print(Key)
    print(Key2)
    print(Key.shape, Key2.shape) 

def weight_resort_V():
    pklname = 'layers_source/transformer.encoder.layers.0.self_attention.query_key_value.pkl'
    query_key_value = in_pickle(pklname).cuda()
    Wt = query_key_value.weight.split([128 * 32, 128 * 2, 128 * 2], dim=0, )[2]
    bias = query_key_value.bias.split([128 * 32, 128 * 2, 128 * 2], dim=0, )[2]

    seq_len = 19
    layernorm_output = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Layer_Norm_output.txt')
    layernorm_output = torch.tensor(layernorm_output).reshape(seq_len, 4096).to(torch.float16).cuda()

    Value = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Value.txt')
    Value = torch.tensor(Value).reshape(seq_len, 128, 2).to(torch.float16).cuda()

    Wt = Wt.reshape(2, 128, 4096).permute(1, 0, 2).reshape(256, 4096)
    # Wt = Wt.reshape(32, 128, 4096).permute(1, 0, 2).reshape(4096, 4096)
    Wt, scales = quant_minmax(Wt.transpose(0, 1), 4, 128)

    out_txt(Wt.to(torch.int32), 'BLOCK00/chatglm_page1/MVM_BN_Wv.txt', is_out=True, is_transpose=True)
    out_txt(scales, 'BLOCK00/chatglm_page1/MVM_BN_Scalev.txt', is_out=True, is_transpose=True)
    out_txt(bias.reshape(2, 128).permute(1, 0), 'BLOCK00/chatglm_page1/MVM_BN_Biasv.txt', is_out=True)

    Wt = Wt.reshape(4096, 128, 2).permute(0, 2, 1).reshape(4096, 256)
    scales = scales.reshape(32, 128, 2).permute(0, 2, 1).reshape(32, 256)
    # Wt = Wt.reshape(4096, 128, 32).permute(0, 2, 1).reshape(4096, 4096)
    # scales = scales.reshape(32, 128, 32).permute(0, 2, 1).reshape(32, 4096)

    qmml = QuantMinMaxLinear(Wt, scales, bias)
    Value2 = qmml(layernorm_output).reshape(seq_len, 2, 128).permute(0, 2, 1)

    print(Value)
    print(Value2)
    print(Value.shape, Value2.shape)

def weight_resort_Dense():
    pklname = 'layers_source/transformer.encoder.layers.0.self_attention.dense.pkl'
    dense = in_pickle(pklname).cuda()
    Wt = dense.weight

    seq_len = 19
    Attention_output = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page2/Attention_output.txt')
    Attention_output = torch.tensor(Attention_output).reshape(1, 32, seq_len, 128).to(torch.float16).cuda()
    Attention_output = Attention_output.permute(2, 0, 1, 3)
    new_context_layer_shape = Attention_output.size()[:-2] + (128 * 32, )
    Attention_output = Attention_output.reshape(*new_context_layer_shape)

    MVM_BN_RES_output = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page3/MVM_BN_RES_output.txt')
    MVM_BN_RES_output = torch.tensor(MVM_BN_RES_output).reshape(seq_len, 4096).to(torch.float16).cuda()

    Wt = Wt.reshape(32, 128, 4096).permute(1, 0, 2).reshape(4096, 4096)
    Wt, scales = quant_minmax(Wt.transpose(0, 1), 4, 128)

    out_txt(Wt.to(torch.int32), 'BLOCK00/chatglm_page3/MVM_BN_RES_weight.txt', is_out=True, is_transpose=True)
    out_txt(scales, 'BLOCK00/chatglm_page3/MVM_BN_RES_scales.txt', is_out=True, is_transpose=True)

    Wt = Wt.reshape(4096, 128, 32).permute(0, 2, 1).reshape(4096, 4096)
    scales = scales.reshape(32, 128, 32).permute(0, 2, 1).reshape(32, 4096)

    qmml = QuantMinMaxLinear(Wt, scales)
    output = qmml(Attention_output).reshape(seq_len, 32, 128).permute(0, 2, 1).reshape(seq_len, 4096)

    Embedded_input = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/chatglm_page1/Embedded_input.txt')
    Embedded_input = torch.tensor(Embedded_input).reshape(seq_len, 4096).to(torch.float16).cuda()

    MVM_BN_RES_output2 = Embedded_input + output

    print(MVM_BN_RES_output)
    print(MVM_BN_RES_output2)
    print(MVM_BN_RES_output.shape, MVM_BN_RES_output2.shape)

def merge_QKV():
    # weight
    MVM_BN_Wq = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Wq.txt')
    MVM_BN_Wq = torch.tensor(MVM_BN_Wq).reshape(128*32, 4096).to(torch.float16).cuda()

    MVM_BN_Wk = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Wk.txt')
    MVM_BN_Wk = torch.tensor(MVM_BN_Wk).reshape(128*2, 4096).to(torch.float16).cuda()

    MVM_BN_Wv = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Wv.txt')
    MVM_BN_Wv = torch.tensor(MVM_BN_Wv).reshape(128*2, 4096).to(torch.float16).cuda()

    MVM_BN_Wqkv = torch.cat((MVM_BN_Wq, MVM_BN_Wk, MVM_BN_Wv), dim=0)
    out_txt(MVM_BN_Wqkv.to(torch.int32), 'BLOCK00/chatglm_page1/MVM_BN_Wqkv.txt', is_out=True)

    # scales
    MVM_BN_Scaleq = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Scaleq.txt')
    MVM_BN_Scaleq = torch.tensor(MVM_BN_Scaleq).reshape(4096, 32).to(torch.float16).cuda()

    MVM_BN_Scalek = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Scalek.txt')
    MVM_BN_Scalek = torch.tensor(MVM_BN_Scalek).reshape(256, 32).to(torch.float16).cuda()

    MVM_BN_Scalev = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Scalev.txt')
    MVM_BN_Scalev = torch.tensor(MVM_BN_Scalev).reshape(256, 32).to(torch.float16).cuda()

    MVM_BN_Scaleqkv = torch.cat((MVM_BN_Scaleq, MVM_BN_Scalek, MVM_BN_Scalev), dim=0)
    out_txt(MVM_BN_Scaleqkv.to(torch.float16), 'BLOCK00/chatglm_page1/MVM_BN_Scaleqkv.txt', is_out=True)

    # bias
    MVM_BN_Biasq = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Biasq.txt')
    MVM_BN_Biasq = torch.tensor(MVM_BN_Biasq).reshape(128, 32).to(torch.float16).cuda()

    MVM_BN_Biask = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Biask.txt')
    MVM_BN_Biask = torch.tensor(MVM_BN_Biask).reshape(128, 2).to(torch.float16).cuda()

    MVM_BN_Biasv = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/MVM_BN_Biasv.txt')
    MVM_BN_Biasv = torch.tensor(MVM_BN_Biasv).reshape(128, 2).to(torch.float16).cuda()

    MVM_BN_Biasqkv = torch.cat((MVM_BN_Biasq, MVM_BN_Biask, MVM_BN_Biasv), dim=1)
    out_txt(MVM_BN_Biasqkv.to(torch.float16), 'BLOCK00/chatglm_page1/MVM_BN_Biasqkv.txt', is_out=True)

    seq_len = 19
    # Query
    Query = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Query.txt')
    Query = torch.tensor(Query).reshape(seq_len, 128, 32).to(torch.float16).cuda()

    Key = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Key.txt')
    Key = torch.tensor(Key).reshape(seq_len, 128, 2).to(torch.float16).cuda()

    Value = get_txt('/home/yuhao408/Myprojects/MyGLM2/out_txt/BLOCK00/chatglm_page1/Value.txt')
    Value = torch.tensor(Value).reshape(seq_len, 128, 2).to(torch.float16).cuda()

    Query_Key_Value = torch.cat((Query, Key, Value), dim=2)
    out_txt(Query_Key_Value.to(torch.float16), 'BLOCK00/chatglm_page1/Query_Key_Value.txt', is_out=True)

if __name__ == "__main__":
    # main()
    # gpqt_layer()

    # weight_resort_QKV()
    # weight_resort_Q()
    # weight_resort_K()
    # weight_resort_V()
    weight_resort_Dense()
    
    # merge_QKV()
