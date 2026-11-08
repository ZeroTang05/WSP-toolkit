"""Python 与 MCP 共用的工具参数、执行入口和结果校验。"""

from collections.abc import Callable
from dataclasses import dataclass

from jsonschema import Draft202012Validator, ValidationError


def object_schema(properties, *required):
    """工具参数只接受明确登记的字段。"""
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


@dataclass(frozen=True)
class ToolSpec:
    """工具只登记一次，两个入口都使用同一个 Python 函数。"""

    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]
    permission: str = "observe"
    output_schema: dict | None = None


class ToolRegistry:
    """按照宿主授予的权限列工具，并在每次调用前校验输入。"""

    def __init__(self, specs, permissions=("observe",)):
        self._tools = {}
        self._permissions = frozenset(permissions)
        for spec in specs:
            if spec.name in self._tools:
                raise ValueError(f"Duplicate tool: {spec.name}")
            Draft202012Validator.check_schema(spec.input_schema)
            result_validator = None
            if spec.output_schema is not None:
                Draft202012Validator.check_schema(spec.output_schema)
                result_validator = Draft202012Validator(spec.output_schema)
            self._tools[spec.name] = (spec, Draft202012Validator(spec.input_schema), result_validator)

    def list_tools(self):
        """返回当前权限可以发现的工具。"""
        return tuple(spec for spec, _, _ in self._tools.values() if spec.permission in self._permissions)

    def call(self, name, arguments):
        """按与发现接口相同的权限执行登记的函数。"""
        spec, validator, result_validator = self._tools[name]
        if spec.permission not in self._permissions:
            raise PermissionError(f"Tool permission required: {spec.permission}")
        validator.validate(arguments)
        result = spec.handler(**arguments)
        if result_validator is not None:
            try:
                result_validator.validate(result)
            except ValidationError as error:
                raise RuntimeError(f"Tool returned an invalid result: {name}") from error
        return result
