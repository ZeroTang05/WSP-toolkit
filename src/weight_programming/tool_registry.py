"""Tool parameters, execution entry points, and result verification shared by Python and MCP."""

from collections.abc import Callable
from dataclasses import dataclass

from jsonschema import Draft202012Validator, ValidationError


def object_schema(properties, *required):
    """Tool parameters only accept explicitly registered fields."""
    return {"type": "object", "properties": properties, "required": list(required),
            "additionalProperties": False}


@dataclass(frozen=True)
class ToolSpec:
    """The tool is only registered once, and both entries use the same Python function."""

    name: str
    description: str
    input_schema: dict
    handler: Callable[..., dict]
    permission: str = "observe"
    output_schema: dict | None = None


class ToolRegistry:
    """List tools according to the permissions granted by the host and verify input before each call."""

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
        """Returns tools discoverable with current permissions."""
        return tuple(spec for spec, _, _ in self._tools.values() if spec.permission in self._permissions)

    def call(self, name, arguments):
        """Execute registered functions with the same permissions as the discovery interface."""
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
