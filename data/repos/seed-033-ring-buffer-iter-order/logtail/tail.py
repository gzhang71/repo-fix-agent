from .ringbuffer import RingBuffer


def tail(lines, n=10):
    """Last n lines, oldest first."""
    buf = RingBuffer(n)
    for line in lines:
        buf.append(line.rstrip("\n"))
    return list(buf)


def tail_matching(lines, needle, n=10):
    return tail((l for l in lines if needle in l), n)
