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


def Attention(q, k, v):
    atten = adr.hbm.mvm_afterTRP(q, k)
    atten = adr.hbm.softmax(atten)
    atten = adr.hbm.mvm_afterF2W(atten, v)
    return atten


def PatchEmbedding(image, prefix):
    wt_prefix = "PatchEmb"
    wt0 = adr.hbm.const_hbm(f"{wt_prefix}_wt0", os.path.join(prefix, ""), [14, 14, 3, 1792])
    cls_emb = adr.hbm.const_ddr(f"{wt_prefix}_cls_emb", os.path.join(prefix, ""), [1, 1, 1792])
    pos_emb = adr.hbm.const_ddr(f"{wt_prefix}_pos_emb", os.path.join(prefix, ""), [1, 1+96*96, 1792])

    tp_out = adr.hbm.conv2d(image, wt0, strides=[14, 14], padding=[0, 0])
    tp_out = adr.reshape(tp_out, [1, 96*96, 1792])
    tp_out = adr.concat([tp_out, cls_emb], dim=1)
    output = adr.hbm.add(tp_out, pos_emb)

    return output


def EVA2CLIPTransformer(embedding, gelu, prefix, index):
    wt_prefix = f"CLIPTf{index}"
    hidden = adr.hbm.var_ddr("hidden", [1, 1+96*96, 1792])
    wt_qkv = adr.hbm.const_hbm(f"{wt_prefix}_wt_qkv", os.path.join(prefix, ""), [1792, 5376])
    bn_qkv = adr.hbm.const_ddr(f"{wt_prefix}_bn_qkv", os.path.join(prefix, ""), [5376*2])
    wt1 = adr.hbm.const_hbm(f"{wt_prefix}_wt1", os.path.join(prefix, ""), [1792, 1792])
    bn1 = adr.hbm.const_ddr(f"{wt_prefix}_bn1", os.path.join(prefix, ""), [1792*2])
    bn2 = adr.hbm.const_ddr(f"{wt_prefix}_bn2", os.path.join(prefix, ""), [1792*2])
    wt3 = adr.hbm.const_hbm(f"{wt_prefix}_wt3", os.path.join(prefix, ""), [1792, 15360])
    bn3 = adr.hbm.const_ddr(f"{wt_prefix}_bn3", os.path.join(prefix, ""), [15360*2])
    wt4 = adr.hbm.const_hbm(f"{wt_prefix}_wt4", os.path.join(prefix, ""), [15360, 1792])
    bn4 = adr.hbm.const_ddr(f"{wt_prefix}_bn4", os.path.join(prefix, ""), [1792*2])
    bn5 = adr.hbm.const_ddr(f"{wt_prefix}_bn5", os.path.join(prefix, ""), [1792*2])

    qkv = adr.hbm.mvm_bn(hidden, wt_qkv, bn_qkv)
    qkv = adr.reshape(qkv, [16*3, 1+96*96, 112])
    qkv = adr.split(qkv, 0, [16, 16, 16])
    q, k, v = qkv[0], qkv[1], qkv[2]
    atten_out = Attention(q, k, v)
    atten_out = adr.reshape(atten_out, [1, 1+96*96, 1792])
    tp_out = adr.hbm.mvm_bn(atten_out, wt1, bn1)
    tp_out = adr.hbm.layer_norm(tp_out, bn2)
    hidden = adr.hbm.add(tp_out, hidden)
    tp_out = adr.hbm.mvm_bn(hidden, wt3, bn3)
    tp_out = adr.hbm.activate(tp_out, gelu)
    tp_out = adr.hbm.mvm_bn(tp_out, wt4, bn4)
    tp_out = adr.hbm.layer_norm(tp_out, bn5)
    output = adr.hbm.add(tp_out, hidden)

    return adr.Function(f"transformer{index}", [embedding], output)


def EVA2CLIPLinearProj(hidden, gelu, silu, prefix):
    wt_prefix = f"CLIPLp"
    wt0 = adr.hbm.const_hbm(f"{wt_prefix}_wt0", os.path.join(prefix, ""), [2, 2, 1792, 1792])
    wt1 = adr.hbm.const_hbm(f"{wt_prefix}_wt1", os.path.join(prefix, ""), [1792, 4096])
    bn2 = adr.hbm.const_ddr(f"{wt_prefix}_bn2", os.path.join(prefix, ""), [4096*2])
    wt3 = adr.hbm.const_hbm(f"{wt_prefix}_wt3", os.path.join(prefix, ""), [4096, 14336])
    wt4 = adr.hbm.const_hbm(f"{wt_prefix}_wt4", os.path.join(prefix, ""), [4096, 14336])
    bn4 = adr.hbm.const_ddr(f"{wt_prefix}_bn4", os.path.join(prefix, ""), [14336*2]) # zero
    wt5 = adr.hbm.const_hbm(f"{wt_prefix}_wt5", os.path.join(prefix, ""), [14336, 4096])
    boi = adr.hbm.const_ddr(f"{wt_prefix}_boi", os.path.join(prefix, ""), [1, 1, 4096])
    eoi = adr.hbm.const_ddr(f"{wt_prefix}_eoi", os.path.join(prefix, ""), [1, 1, 4096])
    # bn5 = adr.hbm.const_ddr(f"{wt_prefix}_bn5", os.path.join(prefix, ""), [1792*2])

    hidden = adr.split(hidden, 1, [1, 96*96])[1]
    hidden = adr.reshape(hidden, [96, 96, 1792])
    tp_out = adr.hbm.conv2d(hidden, wt0, strides=[2, 2])
    tp_out = adr.reshape(tp_out, [1, 48*48, 1792])
    tp_out = adr.hbm.mvm(tp_out, wt1)
    tp_out = adr.hbm.layer_norm(tp_out, bn2)
    tp_out = adr.hbm.activate(tp_out, gelu)
    t0_out = adr.hbm.mvm(tp_out, wt3)
    t0_out = adr.hbm.activate(t0_out, silu)
    tp_out = adr.hbm.mvm_bn_res(tp_out, wt4, bn4, t0_out, res_mul=1)
    tp_out = adr.hbm.mvm(tp_out, wt5)
    output = adr.concat([boi, tp_out, eoi], dim=1)
    return output


def EVA2CLIPModel(prefix="test", device=dlavm.device.HBM0603):
    image = adr.hbm.var_ddr("image", [1344, 1344, 3])
    gelu_wt = adr.hbm.const_ddr("global::gelu_act", os.path.join(prefix, ""), [32*3], adr.DataEnum.int8)
    silu_wt = adr.hbm.const_ddr("global::silu_act", os.path.join(prefix, ""), [32*3], adr.DataEnum.int8)

    hidden = PatchEmbedding(image, prefix)
    for i in range(1):
        hidden = EVA2CLIPTransformer(hidden, gelu_wt, prefix, i)

    hidden = EVA2CLIPLinearProj(hidden, gelu_wt, silu_wt, prefix)
    output = transform.infer_type(hidden, device)
    print(output)
    print(hidden)


if __name__ == "__main__":
    # sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000
    device = dlavm.device.HBM0603

    EVA2CLIPModel(device=device)