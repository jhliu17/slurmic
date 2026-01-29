from dataclasses import dataclass
from slurmic.parser import parse_from_cli


def test_parser():
    @dataclass
    class Args:
        a: int
        b: str
        c: bool

    args = parse_from_cli(Args, parser="tyro", args=["--a", "1", "--b", "test", "--c", "True"])
    print(args)
    assert args.a == 1
    assert args.b == "test"
    assert args.c
