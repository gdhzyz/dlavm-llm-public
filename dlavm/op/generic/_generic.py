from ...adr.base import Op, Tensor, Tuple
from ...utils import tools

'''
TODO: How to support dynamic shape in new_shape attribute
  if convert static graph into this operation
@brief: Reshape op
  @args[0]: input feature data, [x]
  @attrs: new_shape
  @output: Tensor, [x]
'''
def ReshapeRel(args, attrs):
    if len(args) not in [1]:
        return False, "error length arguments! support 1, found " + str(len(args))
    new_shape = attrs["new_shape"]
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    if tools.cumprod(new_shape) != tools.cumprod(dshape):
        return False, "new shape should be matched"
    return True, Tensor(new_shape, dtype, device)

Op.Register("reshape", ReshapeRel)

'''
@brief: Transpose op
  @args[0]: input feature data, [x]
  @attrs: new_axis
  @output: Tensor, [x]
'''
def TransposeRel(args, attrs):
    if len(args) not in [1]:
        return False, "error length arguments! support 1, found " + str(len(args))
    new_axis = attrs["new_axis"]
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    if len(new_axis) != len(dshape):
        return False, "the dim of new axis should be same with data shape"
    new_shape = [dshape[new_axis[i]] for i in len(dshape)]
    return True, Tensor(new_shape, dtype, device)

Op.Register("transpose", TransposeRel)

'''
@brief: Split op
  @args[0]: input feature data, [x]
  @attrs: size, list
  @attrs: axis, int
  @output: Tensor, [x]
'''
def SplitRel(args, attrs):
    if len(args) not in [1]:
        return False, "error length arguments! support 1, found " + str(len(args))
    axis, size = attrs["axis"], attrs["size"]
    device = args[0].device
    dtype = args[0].dtype
    dshape = args[0].shape
    if sum(size) != dshape[axis]:
        return False, "the dim of split data should be same with data shape"
    l_tensor = []
    for n in size:
        new_shape = [n if i == axis else i for i in range(len(dshape))]
        l_tensor.append(Tensor(new_shape, dtype, device))
    return True, Tuple(l_tensor)

Op.Register("split", SplitRel)
