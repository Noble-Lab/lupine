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
    name='lupine',
    version='0.1',
    description="proteomics imputation with lupine",
    long_description=readme + '\n\n' + history,
    author="Lincoln Harris",
    author_email='lincolnh@uw.edu',
    url='https://github.com/Noble-Lab/lupine',
    packages=[
        'lupine',
    ],
    package_dir={'lupine':
                 'lupine'},
    include_package_data=True,
    install_requires=requirements,
    license="MIT",
    zip_safe=False,
    keywords='lupine',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.9',
    ],
    entry_points={
        'console_scripts': [
            'lupine = lupine.commandline:cli'
        ]
    },
    test_suite='tests',
    tests_require=test_requirements
)
