_logs = {}


def LOG_WITH_PREFIX(prefix, log):
    global _logs
    if prefix not in _logs:
        _logs[prefix] = []
    _logs[prefix].append(log)


def LOG_CLEAR():
    global _logs
    _logs = {}


def GET_LOG():
    return _logs


def LOG_EXPORT(path):
    with open(path, "w") as f:
        for i, j in _logs.items():
            f.write(f"------------------------------------------{i}------------------------------------------\n")
            for n in j:
                f.write(n)
                f.write("\n")