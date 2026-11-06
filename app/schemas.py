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

# 工具函数schema
class Defect_Standard_Data(BaseModel):
    """
    定义缺陷查询函数的输入
    """
    defect_type: str = Field(description="缺陷类别，例如划痕、气孔以及裂纹等")

class Sync_Knowledge_Base_Data(BaseModel):
    """
    定义更新知识库函数输入
    """
    scope: Literal["standards", "reports", "all"] = Field(
        default="all",
        description=(
            "指定需要同步/变化预览的知识库范围："
            "standards表示仅同步/变化预览'data/standards'中的质检标准文档"
            "reports表示仅同步/变化预览'data/reports'中的历史检测案例"
            "all表示同时同步/变化预览'data/reports'和'data/standards'中的质检标准文档和历史检测案例"
        )
    )
    mode: Literal["dry_run", "apply"] = Field(
        description=(
            "指定是否进行向量库变更"
            "当用户指定要变更向量库时mode就为apply，如果用户只需要查看向量库的现状，则mode就为dry_run"
            "dry_run 只预览；apply 执行向量库写入。"
            "dry_run表示只扫描并返回知识库是否存在变化"
            "apply表示基于知识库新增、更新或删除向量库"
        )
    )

class Search_Knowledge_Base_Data(BaseModel):
    """
    定义RAG检索函数输入
    """
    query: str = Field(description="用户提问")
    mode: Literal["current", "history"] = Field(
        default="current",
        description=(
            "指定标准文档的查询范围"
            "当用户指定要查询最新标准规范文档时，mode使用current，如果用户指定查询含以往旧版本标准规范文档时，mode使用history"
        )
    )
    top_k: int = Field(
        default=5,
        description="表示当前RAG检索阶段选择最佳的k个chunk，默认为5，不允许修改"
    )