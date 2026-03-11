"""Tests for logging configuration in orion-compiler library code."""

import logging


def test_compiler_logger_name():
    """Verify that compiler.py logger name matches its module."""
    from orion_compiler import compiler

    assert hasattr(compiler, "logger")
    assert compiler.logger.name == "orion_compiler.compiler"
    assert isinstance(compiler.logger, logging.Logger)


def test_packing_logger_name():
    """Verify that packing.py logger name matches its module."""
    from orion_compiler.core import packing

    assert hasattr(packing, "logger")
    assert packing.logger.name == "orion_compiler.core.packing"
    assert isinstance(packing.logger, logging.Logger)


def test_auto_bootstrap_logger_name():
    """Verify that auto_bootstrap.py logger name matches its module."""
    from orion_compiler.core import auto_bootstrap

    assert hasattr(auto_bootstrap, "logger")
    assert auto_bootstrap.logger.name == "orion_compiler.core.auto_bootstrap"
    assert isinstance(auto_bootstrap.logger, logging.Logger)


def test_network_dag_logger_name():
    """Verify that network_dag.py logger name matches its module."""
    from orion_compiler.core import network_dag

    assert hasattr(network_dag, "logger")
    assert network_dag.logger.name == "orion_compiler.core.network_dag"
    assert isinstance(network_dag.logger, logging.Logger)


def test_level_dag_logger_name():
    """Verify that level_dag.py logger name matches its module."""
    from orion_compiler.core import level_dag

    assert hasattr(level_dag, "logger")
    assert level_dag.logger.name == "orion_compiler.core.level_dag"
    assert isinstance(level_dag.logger, logging.Logger)


def test_utils_logger_name():
    """Verify that utils.py logger name matches its module."""
    from orion_compiler.core import utils

    assert hasattr(utils, "logger")
    assert utils.logger.name == "orion_compiler.core.utils"
    assert isinstance(utils.logger, logging.Logger)


def test_no_print_in_library_modules():
    """Verify that library modules use logging, not print().

    This is a basic check that the logger attribute exists in each module
    that previously used print().
    """
    from orion_compiler import compiler
    from orion_compiler.core import auto_bootstrap, level_dag, network_dag, packing, utils

    modules = [compiler, packing, auto_bootstrap, network_dag, level_dag, utils]
    for mod in modules:
        assert hasattr(mod, "logger"), f"{mod.__name__} missing logger attribute"
