# -*- coding: utf-8 -*-

# Import modified 'os' module with LC_LANG 
from .os_utils import os  # noqa: F401

# Python standard library imports
from functools import partial

# 3rd party libraries
import click

# Within-module imports
from lupine.lupine import impute

click.option = partial(click.option, show_default=True)

settings = dict(help_option_names=['-h', '--help'])

@click.group(options_metavar='', subcommand_metavar='<command>',
             context_settings=settings)
def cli():
    """
    TMT proteomics imputation with deep matrix factorization
    """
    pass

cli.add_command(impute, name='impute')

if __name__ == "__main__":
    cli()
