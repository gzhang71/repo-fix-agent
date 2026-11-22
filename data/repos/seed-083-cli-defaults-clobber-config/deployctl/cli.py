import argparse


def build_parser():
    p = argparse.ArgumentParser(prog="deployctl")
    p.add_argument("--config", help="path to a JSON config file")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--debug", action="store_true")
    return p
