def check_list_types(inputs, types)->list:
    if len(inputs) != len(types):
        return []
    tp = [type(n) for n in inputs]
    result = []
    for t in types:
        n = len(result)
        for i in range(len(tp)):
            if tp[i] == t:
                result.append(inputs[i])
        if n == len(result):
            return []
    return result

