import typing
import pydantic

def ensure_type(obj, type_):
    """
    Ensures that the given object is of the given type. Throws a ValueError if it is not.
    """
    if not isinstance(obj, type_):
        raise ValueError(f"Expected type {type_}, got {type(obj)}")
    return obj


def does_dict_match_typed_dict(obj: dict, typed_dict: type[typing.TypedDict]) -> bool:
    validator = pydantic.TypeAdapter(typed_dict)
    try:
        validator.validate_python(obj)
        return True
    except pydantic.ValidationError:
        return False
