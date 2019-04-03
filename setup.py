#!/usr/bin/env python
# -*- coding: utf-8 -*-


try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read().replace('.. :changelog:', '')

with open('requirements.txt') as requirements_file:
    requirements = requirements_file.read()

test_requirements = [
    'pytest', 'coverage', "flake8"
]

setup(
    name='python_boilerplate',
    version='0.1.0',
    description="python boilerplate contains all the boilerplate you need to create a Python package.",
    long_description=readme + '\n\n' + history,
    author="Lincoln Harris",
    author_email='lincoln.harris@czbiohub.org',
    url='https://github.com/lincoln-harris/python_boilerplate',
    packages=[
        'python_boilerplate',
    ],
    package_dir={'python_boilerplate':
                 'python_boilerplate'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='python_boilerplate',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={
        'console_scripts': [
            'python_boilerplate = python_boilerplate.commandline:cli'
        ]
    },
    test_suite='tests',
    tests_require=test_requirements
)
