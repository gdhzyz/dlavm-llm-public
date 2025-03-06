from dlavm import ne
from dlavm.adr import Op, Attrs


@Op.RegisterAttrs("reshape", "compute")
def ReshapeCompute(args, output, attrs):
    output.offset = args[0].offset
    return output


