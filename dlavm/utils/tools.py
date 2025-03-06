from dlavm import ne

def cumprod(l: list)->(int, float):
    result = 1
    for i in l:
        if not isinstance(i, ne.Expr) or i == -1:
            result = result * i
    return result


def Ceil(data0, data1):
    return (data0 + data1 - 1) // data1


def Ceil_Padding(data0, data1):
    return ((data0 + data1 - 1) // data1) * data1


