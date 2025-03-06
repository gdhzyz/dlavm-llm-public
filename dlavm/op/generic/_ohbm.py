from ...adr.base import Op, Tensor, Tuple
from ...utils import tools
from dlavm import ne
from dlavm.device import ohbm_accel

'''
TODO: just support some special case which static shape
 must be matched and -1 used in new_shape to replace 
 dynamic symbol, like input shape is [1, token, 256] and 
 new_shape is [2, -1, 128], the output shape is [2, token, 128]
@brief: Reshape op
  @args[0]: input feature data, [x]
  @attrs: new_shape
  @output: Tensor, [x]
'''
@Op.RegisterAttrs("reshape", "rel", ohbm_accel.OHBM)
def ReshapeRel(args, attrs):
    if len(args) not in [1]:
        return False, "error length arguments! support 1, found " + str(len(args))
    new_shape = attrs["new_shape"]
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    def cumprod_with_dynamic(shape):
        split_shape = [1, 1]
        for n in shape:
            if isinstance(n, ne.Expr):
                split_shape[0] = split_shape[0] * n
            elif n == -1:
                split_shape[0] = -1
            else:
                split_shape[1] = split_shape[1] * n
        return split_shape
    d_size = cumprod_with_dynamic(dshape)
    n_size = cumprod_with_dynamic(new_shape)
    if isinstance(d_size[0], int) and d_size[0] == 1:
        if n_size[0] == -1:
            if d_size[1] % n_size[1]:
                return False, "could not find a integer for -1 new shape"
            d_size[0] = d_size[1] // n_size[1]
            oshape = [i if i != -1 else d_size[0] for i in new_shape]
        else:
            if d_size[1] != n_size[1]:
                return False, "new_shape does not match origin shape"
            oshape = new_shape
    else:
        if d_size[1] != n_size[1]:
            return False, "does not support dynamic case which static shape does not match"
        if not (isinstance(d_size[0], ne.Expr) and n_size[0] == -1):
            return False, "dynamic shape should use -1 support"
        oshape = [i if i != -1 else d_size[0].simplify() for i in new_shape]
    return True, Tensor(oshape, dtype, device)

