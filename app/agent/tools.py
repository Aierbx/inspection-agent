from pydantic import ValidationError
import requests

from app.schemas import Defect_Standard_Data


Defect_Standard_Data_schema = Defect_Standard_Data.model_json_schema()

TOOL_REGISTRY = {}

def register_tool(schema, args_model):
    def decorator(func):
        func_name = schema["function"]["name"]
        TOOL_REGISTRY[func_name] = {
            "schema": schema,
            "function": func,
            "args_model": args_model,
        }

        return func

    return decorator

@register_tool(
    {
        "type": "function",
        "function": {
            "name": "get_defect_standard",
            "description": (
                "这是一个查询指定缺陷的函数"
                "当用户需要了解裂纹、气孔以及划痕等缺陷使调用"
            ),
            "parameters": Defect_Standard_Data_schema
        }
    },
    Defect_Standard_Data
)
async def get_defect_standard(defect_type: str):
    """今天的假工具：返回本地演示数据。"""
    demo_standards = {
        "裂纹": {
            "guidance": "记录缺陷位置、长度和图像证据，建议人工复检。",
            "source": "本地演示数据",
        },
        "气孔": {
            "guidance": "记录数量、尺寸和分布情况，建议结合工艺记录复核。",
            "source": "本地演示数据",
        },
    }

    result = demo_standards.get(defect_type)

    if result is None:
        return {
            "found": False,
            "message": f"暂未收录“{defect_type}”的演示判定依据。",
            "source": "本地演示数据",
        }

    return {
        "found": True,
        "defect_type": defect_type,
        "guidance": result["guidance"],
        "source": result["source"],
        "is_demo": True,
    }

def get_tool_schema():
    """
    获得注册好的所以工具schemas
    """
    return [tool["schema"] for tool in TOOL_REGISTRY.values()]

async def dispatch_tool(function_name, args):

    if function_name not in TOOL_REGISTRY:
        res = {
            "success": False,
            "error": "unknown_tool"
        }
    else:
        try:
            tool = TOOL_REGISTRY[function_name]
            arguments = tool["args_model"].model_validate(args)
            res = await TOOL_REGISTRY[function_name]["function"](**arguments.model_dump())
        except ValidationError:
            res = {
                "success": False,
                "error": "invalid_arguments"
            }
        except requests.exceptions.Timeout:
            res = {
                "success": False,
                "error": "tool_api_timeout"
            }
        except Exception:
            res = {
                "success": False,
                "error": "tool_execution_failed"
            }

    return res