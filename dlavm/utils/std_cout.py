class StdCout:

    def __init__(self):
        self.texts = [[]]

    def __iadd__(self, text):
        self.cout(text)
        return self

    def cout(self, text):
        if isinstance(text, list):
            self.texts[-1] += text
        else:
            self.texts[-1].append(str(text))
            print(f"[\033[1;32mINFO\033[0m]  {text}")
    
    def separator(self, text=""):
        self.texts.append(str(text))
        self.texts.append([])
        print("-"*50)

    def export(self):
        max_len = -1
        for t in self.texts:
            if isinstance(t, str):
                max_len = max(max_len, len(t))
            else:
                for n in t:
                    max_len = max(max_len, len(n))
        max_len += 5
        template = "* {" + f":<{max_len}s" + "}*"
        tp_split = "*-{" + f":-^{max_len}s" + "}*"
        source = ["/*" + "*" * max_len + "\\"]
        for t in self.texts:
            if isinstance(t, list):
                for n in t:
                    source.append(template.format(n))
            elif isinstance(t, str):
                source.append(tp_split.format(f"{t}"))
        source.append("\\*" + "*" * max_len + "/")
        return "\n".join(source)


if __name__ == "__main__":
    cout = StdCout()
    cout.cout("Hello")
    cout.separator("hehe")
    cout.cout("ThankYou")
    print(cout.export())