import typing

import pydantic


def does_dict_match_typed_dict(obj: dict, typed_dict: type[typing.TypedDict]) -> bool:
    validator = pydantic.TypeAdapter(typed_dict)
    try:
        validator.validate_python(obj)
        return True
    except pydantic.ValidationError:
        return False