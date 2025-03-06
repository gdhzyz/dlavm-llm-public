class Module:

    def __init__(self):
        self.node = []

    def append(self, node):
        self.node.append(node)

    def export(self, tab="  "):
        source = []
        for n in self.node:
            source.append(n.export(0, tab))
        return "\n\n".join(source)


class Value:

    def __init__(self, key, value):
        self.key = key
        if isinstance(value, str):
            value = "\"" + value + "\""
        elif isinstance(value, list):
            value = "[" + ", ".join([str(v) for v in value]) + "]"
        self.value = value

    def export(self, tab_numb=0, tab="  "):
        return tab*tab_numb + f"{self.key}: {self.value}"


class List:

    def __init__(self, key) -> None:
        self.key = key
        self.value = []

    def __len__(self):
        return len(self.value)

    def append(self, value):
        self.value.append(value)

    def export(self, tab_numb=0, tab="  "):
        source = [tab*tab_numb + f"{self.key}" + "{"]
        for v in self.value:
            source.append(v.export(tab_numb+1, tab))
        source.append(tab*tab_numb + "}")
        return "\n".join(source)