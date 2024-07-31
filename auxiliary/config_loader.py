#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
auxiliary.config_loader
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This module loads and parses the values found in the
configuration file named 'CONFIG_FILENAME' [config.ini],
setting up a correspondent dictionary for easy access.

:author: Abel
:version: 0.0.2
"""
# Standard library
import configparser
import os
# Application modules
from auxiliary.decorators import log_func

# Define global variables
CONFIG_FILENAME = "demo_config.ini"
PROJECT_NAME = "fedheonn_ensemble"

# Define helper variables
INTEGER = "int_"
STRING = "str_"
FLOAT = "flt_"
BOOLEAN = "bol_"
LIST = "lst_"
PREFIX = "_"


# Define helper functions
def _parse_value(cp: configparser.ConfigParser, section: str, option: str) -> int | str | bool | float | list:
    """
    _parse_value() Returns the proper value type for the given config option, depending on its prefix.

    :param cp: configparser.ConfigParser
    :param section: current section (string)
    :param option: selected option (string)
    :return: parsed value, it may be |int|str|bool|list
    """
    if option.startswith(STRING):
        return cp.get(section, option)
    elif option.startswith(FLOAT):
        return cp.getfloat(section, option)
    elif option.startswith(BOOLEAN):
        return cp.getboolean(section, option)
    elif option.startswith(INTEGER):
        return cp.getint(section, option)
    elif option.startswith(LIST):
        return cp.getlist(section, option)


def _clean_option(x: str) -> str:
    """
    _clean_option() Removes the prefix of the given option string.

    :param x: option (string)
    :return: prefix free option (string)
    """
    return x[x.index(PREFIX) + 1 :]

def _find_config_file(path: str, project_name: str, cfg_name: str):
    """
    Recursive function intended to find the projects parent directory, where the CONFIG_FILE is expected to be found
    :param path: current directory search
    :param project_name: name of the projects parent directory
    :param cfg_name: name of the configuration file
    :return: config file full path or raises FileNotFoundError
    """
    parent_path, curr_path = os.path.split(path)
    if curr_path:
        if curr_path == project_name:
            config_path = os.path.join(path, cfg_name)
            if not os.path.isfile(config_path):
                raise FileNotFoundError(f"Config file not found! ({cfg_name})")
            else:
                return config_path
        else:
            return _find_config_file(parent_path, project_name, cfg_name)
    else:
        raise FileNotFoundError(f"Projects parent directory not found: {parent_path} | {project_name}")

# Define primary function
@log_func
def load_config(config_name: str = CONFIG_FILENAME, project_name: str = PROJECT_NAME) -> dict[str, dict]:
    """
    load_config() Primary function of this module. Loads, reads and parses the values found in the configuration file.

    :param config_name: name of the configuration file. (string) [defaults to CONFIG_FILENAME]
    :param project_name: name of the projects parent directory. (string) [defaults to PROJECT_NAME]
    :return: parsed and cleaned dictionary config-value pairs ({})
    """
    config_path = _find_config_file(os.getcwd(), project_name, config_name)

    config = {}
    # Lambda function to parse the cp.getlist method in order to split and separate list items
    cp = configparser.ConfigParser(converters={'list': lambda x: [i.strip() for i in x.split(',')]})
    cp.read(config_path)
    for section in cp.sections():
        config[section] = {}
        for option in cp.options(section):
            config[section][_clean_option(option)] = _parse_value(cp, section, option)
    return config


if __name__ == "__main__":
    # Testing purposes
    load_config()
