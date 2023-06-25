"""
Usage:

A dataclass/YAML/CLI config system:
- write a @dataclass with your config options
- make sure every option has a default value
- include a `config: str = ""` option in the dataclass.
- write a main function that takes a single argument of the dataclass type
- decorate your main function with @dataclass_cli
- make sure your main function has a docstring.

The config will be loaded from a YAML file specified by the --config option,
and CLI options will override the config file.

Example from running this file:

> python edit/config.py --help

 Usage: config.py [OPTIONS]

 test

╭─ Options
│ --config        TEXT
│ --hi            INTEGER  [default: 1]
│ --bye           TEXT     [default: bye]
│ --help                   Show this message and exit.
╰─
"""

import shlex
import dataclasses
import inspect
import re
import time
from typing import List, Union

import typer
import typer.testing
import yaml


def conf_callback(ctx: typer.Context, param: typer.CallbackParam, value: str) -> str:
    """
    Callback for typer.Option that loads a config file from the first
    argument of a dataclass.

    Based on https://github.com/tiangolo/typer/issues/86#issuecomment-996374166
    """
    if param.name == "config" and value:
        typer.echo(f"Loading config file: {value}")
        try:
            with open(value, "r") as f:
                conf = yaml.safe_load(f)
            ctx.default_map = ctx.default_map or {}
            ctx.default_map.update(conf)
        except Exception as ex:
            raise typer.BadParameter(str(ex))
    return value


def dataclass_cli(func):
    """
    Converts a function taking a dataclass as its first argument into a
    dataclass that can be called via `typer` as a CLI.

    Additionally, the --config option will load a yaml configuration before the
    other arguments.

    Modified from:
    - https://github.com/tiangolo/typer/issues/197

    A couple related issues:
    - https://github.com/tiangolo/typer/issues/153
    - https://github.com/tiangolo/typer/issues/154
    """

    # The dataclass type is the first argument of the function.
    sig = inspect.signature(func)
    param = list(sig.parameters.values())[0]
    cls = param.annotation
    assert dataclasses.is_dataclass(cls)

    def wrapped(**kwargs):
        # Load the config file if specified.
        if kwargs.get("config", "") != "":
            with open(kwargs["config"], "r") as f:
                conf = yaml.safe_load(f)
        else:
            conf = {}

        # CLI options override the config file.
        conf.update(kwargs)

        # Convert back to the original dataclass type.
        arg = cls(**conf)

        # Actually call the entry point function.
        return func(arg)

    # To construct the signature, we remove the first argument (self)
    # from the dataclass __init__ signature.
    signature = inspect.signature(cls.__init__)
    parameters = list(signature.parameters.values())
    if len(parameters) > 0 and parameters[0].name == "self":
        del parameters[0]

    # Add the --config option to the signature.
    # When called through the CLI, we need to set defaults via the YAML file.
    # Otherwise, every field will get overwritten when the YAML is loaded.
    parameters = [p for p in parameters if p.name != "config"] + [
        inspect.Parameter(
            "config",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=typer.Option("", callback=conf_callback, is_eager=True),
        )
    ]

    # The new signature is compatible with the **kwargs argument.
    wrapped.__signature__ = signature.replace(parameters=parameters)

    # The docstring is used for the explainer text in the CLI.
    wrapped.__doc__ = func.__doc__ + "\n" + ""

    return wrapped


# -----------------------------------------------------------------------------


def collect_steps(globals_, func=None):
    """
    Collect all function in the provided namespace that match the pattern:
    step(\\d+)_.*

    For example, step1_foo, step2_bar, etc.

    Then, add a list of these functions to the provided function's docstring.

    Returns a dictionary mapping the step number to the function.
    """
    pattern = re.compile(r"step(\d+)_.*")
    step_dict = {}
    entries = list(globals_.items())
    for k, v in entries:
        match = pattern.match(k)
        if match:
            step_dict[int(match.group(1))] = v

    if func is not None:
        doc = "\n\nStep List:\n\n" + "\n\n".join(
            f"- {i}: {f.__name__}" for i, f in sorted(step_dict.items())
        )
        func.__doc__ += doc
    return step_dict


def run_steps(step_dict, steps: Union[str, int, List[int]], *args, **kwargs):
    """
    Run a list of steps in order.

    -- step_dict: a dictionary mapping step numbers to functions
    -- steps: a list of step numbers to run
    -- *args, **kwargs: passed to each step function

    Steps can be specified as:
    -- a single integer
    -- a list of integers
    -- "all" to run all steps
    -- a comma-separated string list of integers
    """

    if isinstance(steps, int):
        steps = [steps]
    elif isinstance(steps, list):
        for s in steps:
            if not isinstance(s, int):
                raise ValueError(f"Unknown step input {s}")
    elif steps.strip() == "all":
        steps = sorted(step_dict.keys())
    elif isinstance(steps, str):
        steps = [int(s) for s in steps.split(",")]
    else:
        raise ValueError(f"Unknown step input {steps}")

    for s in steps:
        if s not in step_dict:
            raise ValueError(f"Unknown step {s}")
        print("Beginning step", s)
        start = time.time()
        step_dict[s](*args, **kwargs)
        end = time.time()
        print(f"Step {s} took {end - start:.2f} seconds")


# -----------------------------------------------------------------------------


def run(f, raw_args=None):
    if raw_args is None:
        return typer.run(f)
    else:
        app = typer.Typer(add_completion=False)
        app.command()(f)
        args = shlex.split(raw_args)
        return typer.testing.CliRunner(app, args)


# -----------------------------------------------------------------------------


@dataclasses.dataclass
class Test:
    config: str = ""
    hi: int = 1
    bye: str = "bye"


@dataclass_cli
def test_main(c: Test):
    """test"""
    print(c.hi, c.bye)
    return str(c.hi) + c.bye


if __name__ == "__main__":
    # The function can either be called directly using the dataclass
    # parameters:
    assert test_main(hi=2, bye="hello") == "2hello"

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
        yaml.dump({"hi": 3, "bye": "yummy"}, temp_file)
        # Including a config file:
        assert test_main(config=temp_file.name) == "3yummy"
        # CLI options override the config file:
        assert test_main(config=temp_file.name, hi=15) == "15yummy"

    # We can also call directly via the CLI:
    typer.run(test_main)
