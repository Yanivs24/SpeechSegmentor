#!/usr/bin/env python
# encoding: utf-8

# The MIT License (MIT)
#
# Copyright (c) 2017 CNRS
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# AUTHORS
# Hervé BREDIN - http://herve.niderb.fr/


import versioneer
from setuptools import setup, find_packages

setup(
    # replace "mydatabase" by the name of your database
    name='pyannote.db.switchboard',

    # replace "MyDatabase" by the name of your database
    description="SwitchBoard plugin for pyannote-database",

    # replace with your information
    author='Yaniv Sheena',
    author_email='',

    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),

    # replace "MyDatabase" by the new name of MyDatabase directory
    package_data={
        'SwitchBoard': [
            'data/*',
        ],
    },
    include_package_data=True,
    install_requires=[
        'pyannote.database >= 0.11.2',
        'pyannote.parser >= 0.6.5',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 2",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering"
    ],

    # replace MyDatabase by the name of your database (using CamelCase)
    entry_points="""
        [pyannote.database.databases]
        SwitchBoard=SwitchBoard:SwitchBoard
    """
)
