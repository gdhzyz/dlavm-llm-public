from ...adr.base import Op, Tensor, Tuple


'''
@brief: PosEmb op
  @args[0]: input feature data, [Head, token, chin]
  @args[1]: processed data, [x]
  @output: Tensor, [Head, token, chin]
'''
def PosEmbRel(args, attrs):
    if len(args) not in [2]:
        return False, "error length arguments! support 1, found " + str(len(args))
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    if dshape[-1] not in [128]:
        return False, "error head per channels! support 128, found " + str(dshape[-1])
    oshape = [i for i in dshape]
    tensor = Tensor(oshape, dtype, device)
    if hasattr(args[0], "heads"):
        setattr(tensor, "heads", args[0].heads)
    return True, tensor

Op.Register("glm.pos_emb", PosEmbRel)

