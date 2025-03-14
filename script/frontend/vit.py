import sys

sys.path.append("../..")
sys.setrecursionlimit(3000)

import os
import torch
from vit_pytorch import ViT
from dlavm.frontend.torch import DynamoCompiler, transform

os.environ["CUDA_VISIBLE_DEVICES"] = ""


v = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)


dynamo_compiler = DynamoCompiler()

with torch.no_grad():
    v.eval()
    img = torch.randn(1, 3, 256, 256)
    graphs = dynamo_compiler.importer(v, img)

assert len(graphs) == 1
graph = graphs[0]
pattern_list = [transform.simply_fuse]
graphs[0].fuse_ops(pattern_list)

transform.canonicalize(graphs[0])

with open("vit.prototxt", "w") as f:
    print(transform.export_prototxt(graphs[0]), file=f)
