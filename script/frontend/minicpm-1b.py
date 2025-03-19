import sys

sys.path.append("../..")
sys.setrecursionlimit(3000)

from dlavm.frontend.torch import DynamoCompiler, transform

from modelscope import AutoModelForCausalLM, AutoTokenizer
import torch
torch.manual_seed(0)

path = 'openbmb/MiniCPM-1B-sft-bf16'
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map='cuda', trust_remote_code=True)
model.config.use_cache = False

dynamo_compiler = DynamoCompiler()

with torch.no_grad():
    data = torch.tensor([[1 for i in range(1024)]], dtype=torch.int64).to("cuda")
    graphs = dynamo_compiler.importer(model, data)

assert len(graphs) == 1
graph = graphs[0]
pattern_list = [transform.simply_fuse]
graphs[0].fuse_ops(pattern_list)

transform.canonicalize(graphs[0])

with open("minicpm.prototxt", "w") as f:
    print(transform.export_prototxt(graphs[0]), file=f)
