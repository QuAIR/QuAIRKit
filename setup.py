#!/usr/bin/env python
# Copyright (c) 2023 QuAIR team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""
Install library to site-packages
"""

from pathlib import Path

import setuptools

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setuptools.setup(
    name='quairkit',
    version='0.3.0',
    author='QuAIR team.',
    author_email='leizhang116.4@gmail.com',
    description='QuAIRKit is a Python research toolbox for developing quantum computing, quantum information, and quantum machine learning algorithms.',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://www.quairkit.com',
    packages=[
        'quairkit',
        'quairkit.ansatz',
        'quairkit.core',
        'quairkit.core.state',
        'quairkit.core.state.backend',
        'quairkit.core.utils',
        'quairkit.database',
        'quairkit.loss',
        'quairkit.operator',
        'quairkit.operator.channel',
        'quairkit.operator.gate',
        'quairkit.qinfo',
        'quairkit.application',
        'quairkit.application.comb',
        'quairkit.application.locc',
    ],
    install_requires=[
        'torch>=2.0.0',
        'numpy<2.0.0',
        'scipy',
        'matplotlib',
        'pytest-xdist',
        'sphinx>=6.1.3',
        'readthedocs-sphinx-search',
        'sphinx-rtd-theme>=1.3.0',
    ],
    python_requires='>=3.8, <4',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
    ],
)
