from dlavm import ne
from dlavm.adr import Op, Attrs, Tuple
from dlavm.device import ohbm_accel
from ...basic import Tasks, Ceil


@Op.RegisterAttrs("reshape", "compute")
def ReshapeCompute(args, output, attrs):
    output.offset = args[0].offset
    return output


@Tasks.Register("ohbm.gather", ohbm_accel.OHBM)
def Gather(args, output, attrs, device):
    Head, Win, CHin = args[0].shape
    Hin = 1
    CHin_div_LTout = Ceil(CHin, device.L_Tout)
    feature_in_line_stride = device.HBM_1Row_Bytes * Win
    feature_in_surface_stride = device.HBM_1Row_Bytes * Win * Hin
    feature_in_head_stride = device.HBM_1Row_Bytes * Win * Hin * CHin_div_LTout
    setattr(output, "strides", [feature_in_head_stride, feature_in_surface_stride, feature_in_line_stride])
    output.offset = args[0].offset + (args[0].shape[attrs.get("axis")]-len(attrs.get("index"))) * device.HBM_1Row_Bytes
    if attrs.get("index") != [-1] or attrs.get("axis") != -2:
        print("Warning! Please check gather op")


@Op.RegisterAttrs("gather", "compute")
def GatherCompute(args, output, attrs):
    device = args[0].device
    Tasks.Get("ohbm.gather", device)(args, output, attrs, device)
    return output


@Op.RegisterAttrs("split", "compute", ohbm_accel.OHBM)
def SplitCompute(args, output, attrs):
    device = args[0].device
    offset = args[0].offset
    Head, Win, CHin = args[0].shape
    Hin = 1
    CHin_div_LTout = Ceil(CHin, device.L_Tout)
    feature_in_line_stride = device.HBM_1Row_Bytes * Win
    feature_in_surface_stride = device.HBM_1Row_Bytes * Win * Hin
    feature_in_head_stride = device.HBM_1Row_Bytes * Win * Hin * CHin_div_LTout
    strides = [feature_in_head_stride, feature_in_surface_stride, feature_in_line_stride]
    new_tensors = []
    for t in output.tensors:
        setattr(t, "strides", strides)
        t.offset = offset
        new_tensors.append(t)
        offset += t.get_bytesize(dynamic=attrs["dynamic"]) # TODO: CHECK! 实际偏移是否需要重新计算，需进一步考虑
    return Tuple(new_tensors)


@Op.RegisterAttrs("concat", "compute")
def ConcatCompute(args, output, attrs):
    device = args[0].device
    return output