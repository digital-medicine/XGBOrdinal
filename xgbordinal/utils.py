"""
===================================================================
Title:          utils.py
Description:    Outsourced function
Authors:        Fabian Kahl
===================================================================
"""

from collections import OrderedDict

def copy_structure(source):
    if isinstance(source, dict):
        return {key: copy_structure(value) for key, value in source.items()}
    elif isinstance(source, OrderedDict):
        return OrderedDict((key, copy_structure(value)) for key, value in source.items())
    elif isinstance(source, list):
        return []
    else:
        return None