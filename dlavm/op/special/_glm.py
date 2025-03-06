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
    oshape = [i for i in dshape]
    return True, Tensor(oshape, dtype, device)

Op.Register("glm.pos_emb", PosEmbRel)

