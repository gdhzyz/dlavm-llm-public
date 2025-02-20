import os
import torch
from transformers import AutoTokenizer, AutoModel

os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = "../../HuggingFaceModel/chatglm2-6b"
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
model = AutoModel.from_pretrained(model, trust_remote_code=True)
model.config.use_cache = False

from frontend import DynamoCompiler, transform
dynamo_compiler = DynamoCompiler()

with torch.no_grad():
    data = torch.tensor([[1 for i in range(128)]], dtype=torch.int64)
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
pattern_list = [transform.simply_fuse]
graphs[0].fuse_ops(pattern_list)

transform.canonicalize(graphs[0])

with open("chatglm2.prototxt", "w") as f:
    print(transform.export_prototxt(graphs[0]), file=f)
