import sys

sys.path.append("..")
sys.setrecursionlimit(3000)

import os
import argparse
import dlavm
from dlavm import ne
from dlavm import transform
from dlavm import backend
from dlavm.target import targets
import dlavm.device
from dlavm.utils import StdCout
from time import strftime, localtime

from dlavm.llm.chatglm import chatglm_expr_hbm
from dlavm.llm.llama import llama_expr_hbm
from dlavm.llm.qwen2 import qwen2_expr_hbm

config = {
    "chatglm": chatglm_expr_hbm,
    "llama": llama_expr_hbm,
    "qwen2": qwen2_expr_hbm,
}


def main_compile(device, args):
    name = f"{args.model}_{device.MAX_TOKEN}"
    name += "_aux" if args.aux else ""
    name += "_wt2hbm" if args.wt2hbm else ""
    name += "_debug" if args.debug else ""
    name += "_clock" if args.clock else ""
    name += "_onchip" if args.onchip else ""
    name += "_" + strftime('%m%d_%H%M', localtime())

    cout = StdCout()

    cout += f"Compiler target device version: {device.version}"
    cout += "Load model from pre-build compiler-ir"
    output = config[args.model](device, args.prefix, args.debug, ir=True)

    cout += f"Load {args.model} finish with debug {args.debug}"
    cout.separator()
    cout += f"Start model graph optimization"

    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    global_attrs = {"kvcache": kvcache, "last_token": last_token}
    output = transform.infer_type(output, device, attrs=global_attrs)

    cout += f"Start model compiler building..."

    target = targets.hpp
    init_addr = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "insts": "runtime", "hbm": 0x0, "hbm_cache": "hbm", "onchip": 0x0}
    build_config = {"wt2hbm": args.wt2hbm, "debug": True, "ddr_base": 0x20000_0000, "hbm_base": 0x0, "align": 0x4000, "lite": args.lite, "namespace": args.namespace}
    mod = backend.build(output, init_addr, name, args.aux, target, build_config)

    cout += f"Finish model compiler building"
    cout.separator()
    cout += f"Generate code with target {target}"

    source = mod.get_source()
    # source = mod.get_source(name)

    save_path = os.path.join(args.save, name + ".h")
    inst_path = os.path.join(args.save, name + ".bin")
    log_path = os.path.join(args.save, name + ".log")
    with open(save_path, "w") as f:
        f.write(source)
        cout += f"Saved in {save_path}"
    if args.aux:
        cout += f"Generate instructions with aux model"
        inst = mod.get_insts_bin()
        with open(inst_path, "wb") as f:
            f.write(inst)
        cout += f"Saved in {inst_path}"

    cout += f"Generate logs for model and storages"
    dlavm.utils.LOG_WITH_PREFIX("expression", str(output))
    dlavm.utils.LOG_WITH_PREFIX("storage", str(mod.storage))
    dlavm.utils.LOG_EXPORT(log_path)
    cout += f"Saved in {log_path}"


if __name__ == "__main__":
    device = dlavm.device.hbm_accel.HBM0923

    parser = argparse.ArgumentParser()
    parser.add_argument("--py", action="store_true", default=False, help="python backend mode")
    parser.add_argument("--aux", action="store_true", default=False, help="aux cfg backend mode")
    parser.add_argument("--lite", action="store_true", default=False, help="lite cfg backend mode")
    parser.add_argument("--namespace", action="store_true", default=False, help="namespace for codegen")
    parser.add_argument("--debug", action="store_true", default=False, help="debug for one block")
    parser.add_argument("--clock", action="store_true", default=False, help="clock backend mode")
    parser.add_argument("--wt2hbm", action="store_true", default=False, help="wt2hbm backend mode")
    parser.add_argument("--onchip", action="store_true", default=False, help="onchip mode enable")
    parser.add_argument("--prototxt", action="store_true", default=False, help="visualizes for prototxt model")
    parser.add_argument("--model", type=str, default="chatglm", help="set compile model")
    parser.add_argument("--prefix", type=str, default="BLOCK_write_data", help="set datapath prefix")
    parser.add_argument("--save", type=str, default="../output", help="set save path prefix")
    parser.add_argument("--maxtoken", type=int, default=device.MAX_TOKEN, help="set max token")
    args = parser.parse_args()

    if args.model not in config.keys():
        support = ", ".join(config.keys())
        raise RuntimeError(f"no found {args.model} model, support [{support}] models")

    device.MAX_TOKEN = args.maxtoken
    main_compile(device, args)
    log = dlavm.utils.GET_LOG()

