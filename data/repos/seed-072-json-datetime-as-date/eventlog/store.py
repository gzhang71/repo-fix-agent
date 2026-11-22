from .serialize import dumps, loads


def append(stream, event):
    stream.write(dumps(event) + "\n")


def read_all(stream):
    stream.seek(0)
    return [loads(line) for line in stream if line.strip()]
