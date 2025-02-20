import sys
sys.path.append("..")

import os
import dlavm
from dlavm import adr
from dlavm import ne
from dlavm import codegen
from dlavm import transform
import json
import argparse
import numpy as np
from time import strftime, localtime
import dlavm.utils
sys.setrecursionlimit(3000)  # 将默认的递归深度修改为3000

from dlavm.llm.chatglm import chatglm_expr, chatglm_expr_hbm
from dlavm.llm.llama import llama_expr, llama_expr_hbm
from dlavm.llm.qwen2 import qwen2_expr_hbm

config = {
    "chatglm": chatglm_expr,
    "glm0912": chatglm_expr_hbm,
    "llama": llama_expr,
    "llama0912": llama_expr_hbm,
    "qwen2": qwen2_expr_hbm,
}

def main_compile(device, args):
    name = f"{args.model}_{device.MAX_TOKEN}"
    name += "_aux" if args.aux else ""
    name += "_full" if args.full else "_lite"
    name += "_wt2hbm" if args.wt2hbm else ""
    name += "_debug" if args.debug else ""
    name += "_clock" if args.clock else ""
    name += "_onchip" if args.onchip else ""
    name += "_" + strftime('%m%d_%H%M', localtime())

    output = config[args.model](device, args.prefix, args.debug)

    kvcache = ne.Var("kvcache", 1)
    last_token = ne.Var("last_token", device.MAX_TOKEN)
    global_attrs = {"full":args.full, "kvcache": kvcache, "last_token": last_token}
    output = transform.infer_type(output, device, attrs=global_attrs)

    addr_assign_aux = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "cfg": "runtime", "hbm": 0x0, "hbm_cache": "hbm", "onchip": 0x0}
    addr_assign = {"global": 0x0, "weight": "global", "cache": "weight", "runtime": "cache", "hbm": 0x0, "hbm_cache": "hbm", "onchip": 0x0}
    if args.aux:
        if args.wt2hbm:
            expr, source, storage, _, params, _ = codegen.cfg_wt2hbm(output, name, addr_assign_aux, args.onchip)
        else:
            expr, source, storage, _, params, _ = codegen.cfg_head(output, name, addr_assign_aux, args.onchip)
        cmd_path = os.path.join(args.save, name + ".bin")
        with open(cmd_path, "wb") as f:
            f.write(params)
        print(cmd_path)
    elif args.wt2hbm:
        expr, source, storage, _ = codegen.csb_wt2hbm_head(output, name, addr_assign, args.onchip)
    elif args.py:
        expr, source, storage, _ = codegen.csb_python(output, name, addr_assign, args.onchip)
    elif args.clock:
        expr, source, storage, _ = codegen.csb_test_clock_ops(output, name, addr_assign, args.onchip)
    elif args.prototxt:
        expr, source, storage, _ = codegen.visualize_prototxt(output, name, addr_assign, args.onchip)
    else:
        expr, source, storage, _ = codegen.csb_test_head_ops(output, name, addr_assign, args.onchip)

    save_path = os.path.join(args.save, name + ".h")
    if args.py:
        save_path = os.path.join(args.save, name + ".py")
    elif args.prototxt:
        save_path = os.path.join(args.save, name + ".prototxt")
    log_path = os.path.join(args.save, name + ".log")
    with open(save_path, "w") as f:
        f.write(source)
    dlavm.utils.LOG_WITH_PREFIX("expression", str(expr))
    dlavm.utils.LOG_WITH_PREFIX("storage", str(storage))
    dlavm.utils.LOG_EXPORT(log_path)
    print(save_path)
    print(log_path)


if __name__ == "__main__":
    device = dlavm.device.hbm_accel.HBM0923
    print(device.version)

    parser = argparse.ArgumentParser()
    parser.add_argument("--py", action="store_true", default=False, help="python backend mode")
    parser.add_argument("--aux", action="store_true", default=False, help="aux cfg backend mode")
    parser.add_argument("--full", action="store_true", default=False, help="full cfg backend mode")
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
