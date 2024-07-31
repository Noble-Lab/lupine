#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_hello
----------------------------------

Tests for `python_boilerplate` module.
"""

from click.testing import CliRunner


def test_hello():
    from python_boilerplate.hello import hello

    runner = CliRunner()
    result = runner.invoke(hello, input="Rosalind Franklin")

    assert result.exit_code == 0
    assert result.output.count("Hello Rosalind Franklin") == 5


def test_hello_name():
    from python_boilerplate.hello import hello

    runner = CliRunner()
    result = runner.invoke(hello, ["--name", "Rosalind"])

    assert result.exit_code == 0
    assert result.output.count("Hello Rosalind") == 5


def test_hello_count():
    from python_boilerplate.hello import hello

    runner = CliRunner()
    result = runner.invoke(hello, ["--count", "10",
                                   "--name", "Rosalind"])

    assert result.exit_code == 0
    assert result.output.count("Hello Rosalind") == 10
