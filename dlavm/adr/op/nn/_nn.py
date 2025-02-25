from ...base import Op, Tensor, Tuple

def MVMRel(args, attrs):
    if len(args) not in [2, 3, 4]:
        return False, "too more arguments! support [2, 3, 4], found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape, wshape = args[0].shape, args[1].shape
    if dshape[-1] != wshape[1]:
        return False, "weight shape should be [out_channels, in_channels]"
    oshape = [i for i in dshape]
    oshape[-1] = wshape[0]
    if len(args) > 2:
        if args[2].shape[-1] != wshape[0]*2:
            return False, "bn weight shape should be [out_shannels*2]"
    if len(args) > 3:
        if args[3].shape != oshape:
            return False, "res shape should equal to out shape"
    if attrs.get("arg_max", 0):
        arg_max_tensor = Tensor([1, oshape[-2]], dtype, device)
        setattr(arg_max_tensor, "csb_read", 40)
        tensors = Tuple([Tensor(oshape, dtype, device), arg_max_tensor])
        return True, tensors
    return True, Tensor(oshape, dtype, device)

Op.Register("nn.mvm", MVMRel)

Op.Register("nn.norm")
