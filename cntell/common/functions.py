from pydantic import BaseModel


def fieldsof(cls: BaseModel, required=True) -> list:
    return [f.alias for f in cls.__fields__.values() if f.is_required == required]