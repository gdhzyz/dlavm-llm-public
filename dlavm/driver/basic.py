from functools import reduce
import subprocess
from .. import ne
from ..adr import DataEnum, DataType


def CSB_For(expr, tag):
    tag.append([-2, expr])


def CSB_End(expr, tag):
    tag.append([-1, 0, 0])


def CSB_Write(regs, addr, data):
    if data is None:
        regs.append([1, addr, 0])
    elif isinstance(data, ne.Expr):
        regs.append([1, addr, data.simplify()])
    else:
        regs.append([1, addr, data & 0xffffffff])


def CSB_Read(regs, addr, data):
    if data is None:
        regs.append([0, addr, 0])
    elif isinstance(data, ne.Expr):
        regs.append([0, addr, data.simplify()])
    else:
        regs.append([0, addr, data & 0xffffffff])


def Ceil(data0, data1):
    return (data0 + data1 - 1) // data1


def Ceil_Padding(data0, data1):
    return ((data0 + data1 - 1) // data1) * data1


def make_define(define: dict, simulator: str) -> str:
    define_list = []
    for key, value in define.items():
        if simulator == "modelsim":
            tp_str = f"+define+{key}={value}"
        elif simulator == "vivado":
            tp_str = f"--define {key}={value} "
        else:
            raise RuntimeError("Unsupport " + simulator + " for simulation!")
        define_list.append(tp_str)
    define_str = "".join(define_list)
    return "\"" + define_str + "\""


def TestbenchSIM(tb_name: str, define: dict) -> list:
    from .config import template_rtl, tb_sim_path, tb_debug, tb_macro_log, sim_tool
    if tb_debug:
        tb_macro_log.append({"name": tb_name, "testbench": define})
    csb_rtl = []
    define_str = make_define(define, sim_tool)
    cmd = f"make -C {tb_sim_path} TOP_MODULE={tb_name} SIM_DEFINE={define_str}"
    p_rtl = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    out_rtl, rtl_err = p_rtl.communicate()
    saved_out_rtl = out_rtl.decode("utf-8")
    out_rtl = out_rtl.decode("utf-8").replace("# ", "").split("\n")
    for out in out_rtl:
        if "csb_rtl" in out:
            eval(out)
    if len(csb_rtl) == 0:
        raise RuntimeError(saved_out_rtl)
    return csb_rtl


class Tasks:

    memo = {}

    @classmethod
    def Register(cls, op_name, device):
        def _register_task(task):
            if cls.memo.get(device.name) is None:
                cls.memo[device.name] = {}
            if cls.memo[device.name].get(op_name) is None:
                cls.memo[device.name][op_name] = [[task, device.version]]
            elif len(cls.memo[device.name][op_name]) == 1:
                if device.version > cls.memo[device.name][op_name][0][1]:
                    cls.memo[device.name][op_name].append([task, device.version])
                else:
                    cls.memo[device.name][op_name].insert(0, [task, device.version])
            else:
                for i in range(len(cls.memo[device.name][op_name])):
                    if cls.memo[device.name][op_name][i][1] > device.version:
                        cls.memo[device.name][op_name].insert(i, [task, device.version])
                        return
                cls.memo[device.name][op_name].append([task, device.version])
        return _register_task


    @classmethod
    def Get(cls, op_name, device):
        if op_name not in cls.memo[device.name].keys():
            msg = f"no found op \"{op_name}\", please register first"
            raise RuntimeError(msg)
        if len(cls.memo[device.name][op_name]) > 1:
            for i in range(len(cls.memo[device.name][op_name]) - 1):
                if cls.memo[device.name][op_name][i][1] < device.version:
                    if cls.memo[device.name][op_name][i+1][1] > device.version:
                        return cls.memo[device.name][op_name][i][0]
                    elif cls.memo[device.name][op_name][i+1][1] == device.version:
                        return cls.memo[device.name][op_name][i+1][0]
                elif cls.memo[device.name][op_name][i][1] == device.version:
                    return cls.memo[device.name][op_name][i][0]
                else:
                    msg = f"no available task version \"{op_name}\" for {device.version}, please register first"
                    raise RuntimeError(msg)
            return cls.memo[device.name][op_name][i+1][0]
        else:
            if cls.memo[device.name][op_name][0][1] <= device.version:
                return cls.memo[device.name][op_name][0][0]
            else:
                msg = f"no available task version \"{op_name}\" for {device.version}, please register first"
                raise RuntimeError(msg)