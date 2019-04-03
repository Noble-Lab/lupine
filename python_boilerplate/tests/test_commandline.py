#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_commandline
----------------------------------

Tests for `python_boilerplate` module.
"""

from click.testing import CliRunner


def test_cli():
    from python_boilerplate.commandline import cli

    runner = CliRunner()
    result = runner.invoke(cli)

    assert result.exit_code == 0


def test_cli_short_help():
    from python_boilerplate.commandline import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["-h"])

    assert result.exit_code == 0


def test_cli_long_help():
    from python_boilerplate.commandline import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])

    assert result.exit_code == 0

