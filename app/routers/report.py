import re
from collections import Counter

from fastapi import APIRouter, HTTPException, Request
from pydantic import ValidationError

from app.agent.loop import agent_loop
from app.schemas import (
    DetectionResult,
    GenerateResponse,
    ModelCitation,
    ReportModelOutput,
    VerifiedCitation,
)

CITATION_PATTERN = re.compile(r"\[引用:([^\]\r\n]*)\]")

def validate_report_citations(
    draft: str, model_citations: list[ModelCitation], retrieved_sources: list[dict]
) -> tuple[str, list[VerifiedCitation], list[str]]:
    """核验正文中的引用，返回渲染后的正文、已核验来源和待复核片段 ID。"""
    sources_by_id = {
        source["chunk_id"]: source
        for source in retrieved_sources
        if source["citation_eligible"]
    }
    quotes_by_id = {
        citation.chunk_id.strip(): citation.quote for citation in model_citations
    }
    citation_counts = Counter(citation.chunk_id.strip() for citation in model_citations)
    cited_ids = dict.fromkeys(
        chunk_id.strip() for chunk_id in CITATION_PATTERN.findall(draft)
    )

    verified: dict[str, VerifiedCitation] = {}
    unverified = []
    for chunk_id in cited_ids:
        source = sources_by_id.get(chunk_id)
        quote = quotes_by_id.get(chunk_id, "")
        if (
            source is None
            or citation_counts[chunk_id] != 1
            or not quote.strip()
            or quote not in source["text"]
        ):
            unverified.append(chunk_id or "空片段 ID")
            continue

        document_id = source["document_id"].strip()
        section = (source["section_id"] or "").strip() or (source["chapter_id"] or "").strip()
        if not document_id or not section:
            unverified.append(chunk_id)
            continue
        verified[chunk_id] = VerifiedCitation(
            chunk_id=chunk_id, quote=quote, doc_id=document_id, section=section
        )

    unverified.extend(chunk_id for chunk_id in quotes_by_id if chunk_id not in cited_ids)

    def render_reference(match: re.Match[str]) -> str:
        reference = verified.get(match[1].strip())
        if reference is None:
            return "[引用待复核]"
        return f"[来源：{reference.doc_id}｜{reference.section}]"

    draft = CITATION_PATTERN.sub(render_reference, draft)
    return draft, list(verified.values()), unverified

router = APIRouter(prefix="/report", tags=["report"])

@router.post("/generate", response_model=GenerateResponse)
async def generate(req: DetectionResult, request: Request):
    settings = request.app.state.settings

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

    for attempt in range(1, 4):
        reply, steps, retrieved_sources = await agent_loop(
            client=request.app.state.llm_client,
            model_name=request.app.state.model_name,
            messages=messages,
            max_steps=request.app.state.max_steps,
            mode="report"
        )

        try:
            model_output = ReportModelOutput.model_validate_json(reply)
        except ValidationError as exc:
            if attempt == 3:
                raise HTTPException(status_code=502, detail="模型返回的报告格式无效") from exc

            messages.append({
                "role": "user",
                "content": (
                    "上一次报告格式校验失败。请根据以下错误重新输出完整 JSON：\n"
                    f"{exc.errors(include_input=False, include_url=False)}"
                ),
            })
        else:
            break

    draft, citations, unverified = validate_report_citations(
        model_output.draft, model_output.citations, retrieved_sources
    )
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

    has_pending_judgment = any(
        item.judgment == "待复核" for item in model_output.defect_judgments
    )
    status = (
        "needs_review"
        if defects_requiring_reinspection or unverified or has_pending_judgment
        else "draft"
    )

    return GenerateResponse(draft=draft, citations=citations, status=status)
