try:
    from collections.abc import Mapping
except ImportError:
    from collections import Mapping

import six

from .reducers import tuple_reducer, path_reducer, dot_reducer, underscore_reducer
from .splitters import tuple_splitter, path_splitter, dot_splitter, underscore_splitter


REDUCER_DICT = {
    "tuple": tuple_reducer,
    "path": path_reducer,
    "dot": dot_reducer,
    "underscore": underscore_reducer,
}

SPLITTER_DICT = {
    "tuple": tuple_splitter,
    "path": path_splitter,
    "dot": dot_splitter,
    "underscore": underscore_splitter,
}


def flatten(
    d,
    reducer="tuple",
    inverse=False,
    max_flatten_depth=None,
    enumerate_types=(),
    keep_empty_types=(),
):
    
    enumerate_types = tuple(enumerate_types)
    flattenable_types = (Mapping,) + enumerate_types
    if not isinstance(d, flattenable_types):
        raise ValueError(
            "argument type %s is not in the flattenalbe types %s"
            % (type(d), flattenable_types)
        )

    
    if max_flatten_depth is not None and max_flatten_depth < 1:
        raise ValueError("max_flatten_depth should not be less than 1.")

    if isinstance(reducer, str):
        reducer = REDUCER_DICT[reducer]
    flat_dict = {}

    def _flatten(_d, depth, parent=None):
        key_value_iterable = (
            enumerate(_d) if isinstance(_d, enumerate_types) else six.viewitems(_d)
        )
        has_item = False
        for key, value in key_value_iterable:
            has_item = True
            flat_key = reducer(parent, key)
            if isinstance(value, flattenable_types) and (
                max_flatten_depth is None or depth < max_flatten_depth
            ):
                
                has_child = _flatten(value, depth=depth + 1, parent=flat_key)
                if has_child or not isinstance(value, keep_empty_types):
                    
                    
                    continue

            
            if inverse:
                flat_key, value = value, flat_key
            if flat_key in flat_dict:
                raise ValueError("duplicated key '{}'".format(flat_key))
            flat_dict[flat_key] = value

        return has_item

    _flatten(d, depth=1)
    return flat_dict


def nested_set_dict(d, keys, value):
    
    assert keys
    key = keys[0]
    if len(keys) == 1:
        if key in d:
            raise ValueError("duplicated key '{}'".format(key))
        d[key] = value
        return
    d = d.setdefault(key, {})
    nested_set_dict(d, keys[1:], value)


def unflatten(d, splitter="tuple", inverse=False):
    
    if isinstance(splitter, str):
        splitter = SPLITTER_DICT[splitter]

    unflattened_dict = {}
    for flat_key, value in six.viewitems(d):
        if inverse:
            flat_key, value = value, flat_key
        key_tuple = splitter(flat_key)
        nested_set_dict(unflattened_dict, key_tuple, value)

    return unflattened_dict
