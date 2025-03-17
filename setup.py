# -*- coding: utf-8 -*-
from setuptools import setup

packages = \
['orion',
 'orion.backend',
 'orion.backend.lattigo',
 'orion.backend.python',
 'orion.core',
 'orion.models',
 'orion.nn']

package_data = \
{'': ['*'], 'orion.backend': ['heaan/*', 'openfhe/*']}

install_requires = \
['PyYAML>=6.0',
 'h5py>=3.5.0',
 'matplotlib>=3.1.0',
 'numpy>=1.21.0',
 'scipy>=1.7.0,<=1.14.1',
 'torch>=2.2.0',
 'torchvision>=0.17.0',
 'tqdm>=4.30.0']

setup_kwargs = {
    'name': 'orion-fhe',
    'version': '0.1.20',
    'description': 'A Fully Homomorphic Encryption Framework for Deep Learning',
    'long_description': '# Orion\n\nAdding installation instructions and additional examples shortly.',
    'author': 'Austin Ebel',
    'author_email': 'abe5240@nyu.edu',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.9',
}
from tools.build_lattigo import *
build(setup_kwargs)

setup(**setup_kwargs)
