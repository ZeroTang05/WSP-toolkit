"""用 MCP 的标准输入输出传输接入共用工具登记表。"""

import argparse
from contextlib import redirect_stdout
from functools import partial
import json
from pathlib import Path
import sys

import anyio
from jsonschema import ValidationError
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from .inspection import open_inspection
from .forward_service import PracticeSet
from .patches import ParentMismatch


def create_server(registry):
    """只负责 MCP 消息转换，具体执行仍由登记表负责。"""
    async def list_tools(_context, _params):
        return ListToolsResult(tools=[Tool(name=spec.name, description=spec.description,
                                          input_schema=spec.input_schema, output_schema=spec.output_schema)
                                     for spec in registry.list_tools()])

    async def call_tool(_context, params):
        arguments = params.arguments if params.arguments is not None else {}
        try:
            result = await anyio.to_thread.run_sync(partial(registry.call, params.name, arguments))
        except ValidationError as error:
            return failure("INVALID_ARGUMENTS", error.message)
        except PermissionError as error:
            return failure("PERMISSION_DENIED", str(error))
        except ParentMismatch as error:
            return failure("PARENT_MISMATCH", str(error))
        except FileExistsError as error:
            return failure("ALREADY_EXISTS", str(error))
        except (KeyError, FileNotFoundError):
            return failure("NOT_FOUND", "Tool or registered artifact not found")
        except ValueError as error:
            return failure("INVALID_REQUEST", str(error))
        return CallToolResult(structured_content=result,
                              content=[TextContent(type="text", text=json.dumps(result, allow_nan=False))])

    return Server("wsp-toolkit", on_list_tools=list_tools, on_call_tool=call_tool)


def failure(code, message):
    """让客户端收到可处理的错误原因。"""
    return CallToolResult(is_error=True, structured_content={"code": code, "message": message},
                          content=[TextContent(type="text", text=message)])


async def serve(server):
    """通过官方 SDK 监听本地客户端。"""
    async with stdio_server() as (reader, writer):
        await server.run(reader, writer, server.create_initialization_options())


def load_cases(path, role, name):
    """读取由宿主登记的训练集或开发集。"""
    if path is None:
        return None
    with path.open(encoding="utf-8") as stream:
        return PracticeSet([json.loads(line) for line in stream], role=role, name=name)


def main():
    """从命令行加载宿主指定的真实模型。"""
    parser = argparse.ArgumentParser(description="WSP-toolkit local MCP inspection server")
    parser.add_argument("--model", required=True, help="Local checkpoint directory or Hub repository")
    parser.add_argument("--revision", help="Full Hub commit SHA for a Hub repository")
    parser.add_argument("--workspace", required=True, help="Output directory for this workspace")
    parser.add_argument("--device", default="cpu", help="CPU or a single CUDA device")
    parser.add_argument("--allow-edits", action="store_true", help="Enable patch writes and checkpoint exports")
    parser.add_argument("--allow-interventions", action="store_true", help="Enable temporary activation intervention")
    parser.add_argument("--allow-gradients", action="store_true", help="Enable gradients through the frozen model")
    parser.add_argument("--development-jsonl", type=Path, help="Host-authorized development cases with id, problem and answer")
    parser.add_argument("--dataset-id", default="development", help="Name for the registered development dataset")
    parser.add_argument("--training-jsonl", type=Path, help="Host-authorized training cases with id and problem")
    parser.add_argument("--training-dataset-id", default="training", help="Name for the registered training dataset")
    parser.add_argument("--allow-training", action="store_true", help="Enable adapter training on the registered training set")
    parser.add_argument("--allow-python", action="store_true", help="Enable the Linux bubblewrap CPU Python tool")
    parser.add_argument("--kernel-python", type=Path, help="Optional virtualenv Python for the isolated kernel")
    parser.add_argument("--fact-editor-python", type=Path, help="Python executable in the separate EasyEdit environment")
    parser.add_argument("--fact-editor-source", type=Path, help="Source checkout of EasyEdit")
    parser.add_argument("--comparison-model", type=Path, help="Optional second local checkpoint for comparison")
    args = parser.parse_args()
    if args.allow_training and args.training_jsonl is None:
        parser.error("--allow-training requires --training-jsonl")
    if (args.fact_editor_python is None) != (args.fact_editor_source is None):
        parser.error("Provide both --fact-editor-python and --fact-editor-source")
    if args.fact_editor_python is not None and not (args.allow_edits and args.allow_gradients):
        parser.error("Fact editing requires --allow-edits and --allow-gradients")
    practice = load_cases(args.development_jsonl, "development", args.dataset_id)
    training_data = load_cases(args.training_jsonl, "train", args.training_dataset_id)
    with redirect_stdout(sys.stderr):
        comparison = None
        if args.comparison_model is not None:
            comparison = open_inspection(args.comparison_model, Path(args.workspace) / "comparison",
                                         device=args.device, practice=practice)
        tools = open_inspection(args.model, args.workspace, device=args.device, revision=args.revision,
                                allow_edits=args.allow_edits, allow_interventions=args.allow_interventions,
                                allow_gradients=args.allow_gradients, practice=practice, comparison=comparison,
                                training_data=training_data, allow_training=args.allow_training,
                                allow_python=args.allow_python, kernel_python=args.kernel_python,
                                fact_editor_python=args.fact_editor_python, fact_editor_source=args.fact_editor_source)
    with tools:
        anyio.run(serve, create_server(tools.registry))


if __name__ == "__main__":
    main()
