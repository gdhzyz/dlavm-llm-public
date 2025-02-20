from .sparse_conv import RunConv


def Conv2d(args, output, attrs):
    return RunConv(args[0], args[1], output, attrs)


def Pool(args, output, attrs):
    pass