from fastapi import APIRouter, Request
import re

from app.schemas import GenerateResponse, DetectionResult
from app.agent.loop import agent_loop
from app.config import Settings

CITATION_PATTERN = re.compile(r"\[来源：([^｜\]\r\n]+)｜([^\]\r\n]+)\]")
settings = Settings()

def validate_report_citations(
    draft: str, retrieved_sources: list[dict]
) -> tuple[list[str], list[str]]:
    """
    通过正则匹配草稿文档中实际引用的文档
    """
    allowed = set()
    for source in retrieved_sources:
        section = source["section_id"] or source["chapter_id"]
        if source["citation_eligible"] and section:
            allowed.add((source["document_id"].strip(), section.strip()))

    references = dict.fromkeys(
        (document_id.strip(), section.strip())
        for document_id, section in CITATION_PATTERN.findall(draft)
    )

    citations = []
    unverified_citations = []
    for document_id, section in references:
        reference = f"{document_id}｜{section}"
        if (document_id, section) in allowed:
            citations.append(reference)
        else:
            unverified_citations.append(reference)

    return citations, unverified_citations

router = APIRouter(prefix="/report", tags=["report"])

@router.post("/generate", response_model=GenerateResponse)
async def generate(req: DetectionResult, request: Request):

    defects_requiring_reinspection = [
        defect
        for defect in req.defect_result
        if defect.min_confidence < settings.REINSPECTION_CONFIDENCE_THRESHOLD
    ]

    messages = [
        {
            "role": "user",
            "content": req.model_dump_json()
        }
    ]
    draft, steps, citations = await agent_loop(
        client=request.app.state.llm_client,
        model_name=request.app.state.model_name,
        messages=messages,
        max_steps=request.app.state.max_steps,
        mode="report"
    )
    citations, unverified = validate_report_citations(draft, citations)
    if unverified:
        draft += "\n\n**引用待复核**\n\n"
        draft += "\n".join(f"- {source}" for source in unverified)

    # 添加confidence低于REINSPECTION_CONFIDENCE_THRESHOLD的复检
    marker = "<!-- REINSPECTION_ITEMS -->"

    reinspection_text = "\n".join(
        f"- **建议人工复检**：{defect.defect_type}，"
        f"最低识别置信度为 {defect.min_confidence:.2f}。"
        for defect in defects_requiring_reinspection
    )

    if marker in draft:
        draft = draft.replace(marker, reinspection_text, 1)
    elif reinspection_text:
        draft += f"\n\n## 人工复检\n\n{reinspection_text}\n"

    return GenerateResponse(draft=draft, citations=citations)