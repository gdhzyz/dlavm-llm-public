from .aux_build import AuxBuild
from .regs_build import RegsBuild
from .build_module import BuildModule


def build(expr, init_addr, name, aux, target, config):
    if aux:
        lib, graph, storage, params, insts = AuxBuild(**config).build(expr, init_addr, name)
    else:
        lib, graph, storage, params, insts = RegsBuild(**config).build(expr, init_addr, name)
    mod = BuildModule(lib, graph, storage, params, insts, target)
    return mod
