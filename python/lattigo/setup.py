from setuptools import setup

packages = ["lattigo"]

package_data = {"": ["*"]}

setup_kwargs = {
    "name": "lattigo",
    "version": "0.1.0",
    "description": "Python bindings for Lattigo CKKS primitives",
    "long_description": "None",
    "author": "None",
    "author_email": "None",
    "maintainer": "None",
    "maintainer_email": "None",
    "url": "None",
    "packages": packages,
    "package_data": package_data,
    "python_requires": ">=3.11,<3.13",
}
from build_bridge import *

build(setup_kwargs)

setup(**setup_kwargs)
