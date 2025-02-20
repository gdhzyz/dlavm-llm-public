from ..adr import Op, Tensor
from .. import ne


def to_string(inputs: dict) -> dict :
    outputs = {}
    for key, value in inputs.items():
        if isinstance(value, ne.Expr):
            outputs[key] = str(value.simplify())
        else:
            outputs[key] = str(value)
    return outputs


def CacheDriver(args, output, attrs):
    dshape, wshape, oshape, device = args[0][0].shape, args[1][0].shape, output[0].shape, output[0].device
    ddtype, wdtype, odtype = args[0][0].dtype, args[1][0].dtype, output[0].dtype
    daddrs, waddrs, oaddrs = args[0][1], args[1][1], output[1]
    dshape_NHWT, wshape_NHWT, oshape_NHWT = HWC2NHWT(dshape), HWC2NHWT(wshape), HWC2NHWT(oshape)
    doffset = dshape_NHWT[2] * dshape_NHWT[3] * ddtype.get_bytes()
    woffset = wshape_NHWT[2] * wshape_NHWT[3] * wdtype.get_bytes()
    ooffset = oshape_NHWT[2] * oshape_NHWT[3] * odtype.get_bytes()
    memcpy = []
    memcpy_format = "memcpy((void*)%(oaddrs)s, (void*)%(daddrs)s, %(size)s);"
    for n in range(dshape_NHWT[0]):
        memcpy.append(memcpy_format % to_string({
                                    "oaddrs": oaddrs + ooffset*n, 
                                    "daddrs": waddrs + woffset*n, 
                                    "size": woffset
                                }))
        memcpy.append(memcpy_format % to_string({
                                    "oaddrs": oaddrs + ooffset*n + woffset, 
                                    "daddrs": daddrs + doffset*n, 
                                    "size": doffset
                                }))
    memcpy.append(memcpy_format % to_string({
                                "oaddrs": waddrs, 
                                "daddrs": oaddrs, 
                                "size": (output[0])
                            }))
    return memcpy


Op.Get("cpu.cache").attrs["source"] = CacheDriver
