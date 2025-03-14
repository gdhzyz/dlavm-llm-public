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


def RegsCheckSame(serial1: dict, serial2: dict, ignore_addr: list) -> bool:
    if len(serial1) != len(serial2):
        print(f"*Regs Check Same Error* : length not same, got {len(serial1)} and {len(serial2)}")
        return False
    items1 = list(serial1.items())
    items2 = list(serial2.items())
    match = True
    for i in range(len(items1)):
        if items1[i][0] != items2[i][0]:
            print(f"*Regs Check Same Error* : key of num.{i} not same, got {items1[i][0]} and {items2[i][0]}")
            return False
        if len(items1[i][1]) != len(items2[i][1]):
            print(f"*Regs Check Same Error* : the value length of {items1[i][0]} not same, got {len(items1[i][1])} and {len(items2[i][1])}")
            match = False
            continue
        for n in range(len(items1[i][1])):
            v1 = items1[i][1][n]
            v2 = items2[i][1][n]
            if v1[0] != v2[0] or v1[1] != v2[1]:
                print(f"*Regs Check Same Error* : the regs of {items1[i][0]} not same, got {v1} and {v2}")
                match = False
            if v1[1] not in ignore_addr:
                if v1[0] == 0:
                    if isinstance(v1[2], ne.Expr) or isinstance(v2[2], ne.Expr):
                        print(f"# *Regs Check Same Wranning* : found symbol expression, please check!")
                        str1 = "[" + ", ".join([str(_i) for _i in v1]) + "]"
                        str2 = "[" + ", ".join([str(_i) for _i in v2]) + "]"
                        print(f"# {items1[i][0]}: {str1}, {str2}")
                        continue
                    if v1[2] != v2[2]:
                        print(f"*Regs Check Same Error* : the regs of {items1[i][0]} not same, got {v1} and {v2}")
                        match = False
    return match
