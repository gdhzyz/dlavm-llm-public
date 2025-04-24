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


def RegsCheckAddrOffset(addr_dict: dict) -> bool:
    match = True
    for addr, datas in addr_dict.items():
        v1, v2 = datas
        o1, o2 = [], []
        for i in range(len(v1)-1):
            o1.append(v1[i+1]-v1[i])
            o2.append(v2[i+1]-v2[i])
        if o1 != o2:
            print(f"*Addr Check Same Error* : offset of addr reg {addr} not same, got {o1} and {o2}")
            match = False
    if not match:
        print(addr_dict)
    return match


def RegsCheckSameList(serial1: list, serial2: list, ignore_addr: list) -> bool:
    if len(serial1) != len(serial2):
        print(f"*Regs Check Same Error* : length not same, got {len(serial1)} and {len(serial2)}")
        return False
    match = True
    addr_dict = {}
    for addr in ignore_addr:
        addr_dict[addr] = [[], []]
    for n in range(len(serial1)):
        v1 = serial1[n]
        v2 = serial2[n]
        if v1[0] != v2[0] or v1[1] != v2[1]:
            print(f"*Regs Check Same Error* : the regs not same, got {v1} and {v2}")
            match = False
            str1 = "[" + ", ".join([str(_i) for _i in v1]) + "]"
            str2 = "[" + ", ".join([str(_i) for _i in v2]) + "]"
            print(f"# {str1}, {str2}")
            continue
        if v1[1] not in ignore_addr:
            if v1[0] == 0:
                if isinstance(v1[2], ne.Expr) or isinstance(v2[2], ne.Expr):
                    print(f"# *Regs Check Same Wranning* : found symbol expression, please check!")
                    str1 = "[" + ", ".join([str(_i) for _i in v1]) + "]"
                    str2 = "[" + ", ".join([str(_i) for _i in v2]) + "]"
                    print(f"# {str1}, {str2}")
                    continue
                if v1[2] != v2[2]:
                    print(f"*Regs Check Same Error* : the regs not same, got {v1} and {v2}")
                    match = False
        else:
            addr_dict[v1[1]][0].append(v1[2])
            addr_dict[v1[1]][1].append(v2[2])
    addr_match = RegsCheckAddrOffset(addr_dict)
    return match and addr_match
