from pydantic import BaseModel, Field
from typing import Literal
from datetime import datetime

# 消息schema
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class AgentRequest(BaseModel):
    messages: list[Message]
    session_id: str | None = None  # Day 3 后用于多轮记忆

class ChatResponse(BaseModel):
    reply: str
    steps: int    # agent 实际走了几步，便于你观察行为

# 检测算法返回的结构化结果
class Quality_Metrics(BaseModel):
    """
    定义质量指标
    """
    inspected_count: int = Field(ge=0, description="检测产品数")
    qualified_count: int = Field(ge=0, default=0, description="合格产品数")
    unqualified_count: int = Field(ge=0, default=0, description="不合格产品数")
    defect_rate: float = Field(ge=0.0, le=1.0, default=0, description="不良率")
    historical_defect_rate: float = Field(ge=0.0, le=1.0, default=0, description="历史不良率")
    alert_threshold: float = Field(ge=0.0, le=1.0, default=0, description="报警阈值")

class Defect(BaseModel):
    """
    定义检测结果-某类缺陷的检测信息
    """
    defect_type: str = Field(description="缺陷类别编码")
    # defect_location: str = Field(description="缺陷在产品表面的位置")
    defect_description: str = Field(description="缺陷描述")
    defect_count: int = Field(ge=0, description="存在该类缺陷的数量")
    affected_product_count: int = Field(ge=0, description="受影响的产品数量")
    avg_confidence: float = Field(ge=0.0, le=1.0, description="该类缺陷的平均识别置信度")
    min_confidence: float = Field(ge=0.0, le=1.0, description="该类缺陷的最低识别置信度")
    max_confidence: float = Field(ge=0.0, le=1.0, description="该类缺陷的最高识别置信度")
    evidence_refs: list[str] = Field(description="该类缺陷的图像路径")


class DetectionResult(BaseModel):
    """
    定义检测结构化结果
    """
    schema_version: str = Field(description="检测算法结构化输出JSON版本")
    event_id: str = Field(description="质量异常事件ID")
    production_line_id: str = Field(description="异常流水线ID")
    product_model: str = Field(description="产品型号ID")
    batch_id: str = Field(description="异常事件产品批次ID")
    inspection_device_id: str = Field(description="检测设备ID")
    triggered_at: datetime = Field(description="触发异常时间")

    quality_metrics: Quality_Metrics
    defect_result: list[Defect]

#