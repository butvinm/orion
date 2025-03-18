#!/bin/bash
set -e

# Set Python version from environment variable or use default
PYTHON_VERSION=${PYTHON_VERSION:-"3.10"}
PYTHON_VERSION_NO_DOT=$(echo $PYTHON_VERSION | tr -d ".")

# Find correct Python bin directory
PYBIN=/opt/python/cp${PYTHON_VERSION_NO_DOT}*/bin

echo "Building with Python $PYTHON_VERSION (using $PYBIN)"

# Install build dependencies (especially for Python 3.12)
#$PYBIN/pip install setuptools>=61.0 wheel

# Install uv
$PYBIN/pip install uv

# Build package
$PYBIN/python setup.py bdist_wheel
$PYBIN/uv build --sdist --wheel --out-dir dist

echo "Building manylinux wheels with auditwheel..."

# Repair wheels with auditwheel
/opt/python/cp310-cp310/bin/auditwheel repair dist/*.whl -w dist/manylinux/

# Remove original wheels and replace with manylinux ones
find dist -name "*.whl" -not -path "*/manylinux/*" -delete
mkdir -p dist/
mv dist/manylinux/* dist/ 2>/dev/null || true
rmdir dist/manylinux/ 2>/dev/null || true

echo "Build complete! Manylinux wheels available in dist/"