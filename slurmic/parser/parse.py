import tyro

from typing import Union, Callable, Type, Any


def parse_from_cli(
    ArgsType: Type[Any],
    parser: Union[str, Callable] = "tyro",
    *args,
    **kwargs,
) -> Any:
    # parse with built-in parser or custom parser function
    parser_fn = None
    if isinstance(parser, str):
        if parser == "tyro":
            parser_fn = tyro.cli
        else:
            raise ValueError(f"Parser `{parser}` is not supported.")
    else:
        parser_fn = parser
    args: ArgsType = parser_fn(ArgsType, *args, **kwargs)
    return args
