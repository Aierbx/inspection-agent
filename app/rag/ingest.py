import hashlib
from pathlib import Path
import json
from json import JSONDecodeError
import chromadb
import re
from sentence_transformers import SentenceTransformer

from app.config import Settings, BASE_DIR

"""
sync_knowledge_base()        # 总入口：预览或执行知识库同步
collect_source_files()       # 扫描 standards / reports 中的 Markdown 文件
load_manifest()              # 读取 index_manifest.json
load_catalog()               # 读取 catalog.json，获取当前手册版本关系
calculate_content_hash()     # 计算源文件内容 hash
parse_document()             # 解析 front matter 和 Markdown 正文
is_citation_eligible()       # 判断文档当前是否允许被引用
split_markdown()             # 按章节切分 Markdown，生成知识块
build_change_plan()          # 对比源文件和 manifest，生成新增/变更/删除计划
apply_change_plan()          # 根据变更计划更新或删除 Chroma 向量
save_manifest()              # 保存最新 index_manifest.json
get_or_create_collection()   # 创建或打开 Chroma 持久化 collection
"""

settings = Settings()

embedding_model = None

def get_embedding_model() -> SentenceTransformer:
    """
    加载embedding模型
    """
    global embedding_model

    if embedding_model is None:
        embedding_model = SentenceTransformer(
            model_name_or_path=settings.EMBEDDING_MODEL_NAME,
            cache_folder=str(settings.EMBEDDING_MODEL_DIR),
            device="cpu",
        )

    return embedding_model

def create_embeddings(texts: list[str]) -> list[list[float]]:
    """
    文本向量化
    """
    model = get_embedding_model()

    vetors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    return vetors.tolist()

def get_embedding_config() -> dict:
    return {
        "provider": "sentence_transformers",
        "model_name": settings.EMBEDDING_MODEL_NAME,
        "normalize_embeddings": True,
    }

async def collect_source_files(
        scope: str,
):
    """
    获取知识库文件路径
    """
    source_dirs = {
        "standards": settings.STANDARDS_DIR,
        "reports": settings.REPORTS_DIR,
    }

    if scope == "all":
        selected_scopes = ["standards", "reports"]
    else:
        selected_scopes = [scope]

    files: list[Path] = []
    for selected_scope in selected_scopes:
        folder = source_dirs[selected_scope]

        if folder.exists():
            files.extend(folder.rglob("*.md"))

    return files

def create_empty_manifest() -> dict:
    """
    新建清单字典，清单结构如下：
    {
        "schema": 1,                    # manifest结构的版本
        "embedding": {
            ”provider“      # embedding模型来源库
            "model_name"    # 模型名称
            "normalize_embeddings"  # 是否进行标准化
        },                # embedding模型信息
        "catalog": {                    # 上次同步时catalog.json对应的SHA-256
            "content_sha256": None
        },
        "documents": {                  # 每个已入库的文件索引记录
            data_path:{
                "document_id"           # 文件id
                "document_type"         # 文件类型
                "content_sha256"        # 正文的SHA-256
                "version_status"        # 文档版本
                "citation_eligible"     # 是否可被引用
                "chunk_ids"             # 该文件有哪些chunk，对应的chunk_id是哪些
            }

        },
    }
    """
    return {
        "schema": 2,                    # manifest结构的版本
        "embedding": {
            "provider": None,
            "model_name": None,
            "normalize_embeddings": False,
        },                # embedding模型信息
        "catalog": {                    # 上次同步时catalog.json对应的SHA-256
            "content_sha256": None
        },
        "documents": {},                 # 每个已入库的文件索引记录
    }

def load_manifest():
    """
    加载清单json
    """
    manifest_path = settings.INDEX_MANIFEST_DIR

    if not manifest_path.exists():
        return create_empty_manifest()

    try:
        with open(manifest_path / "manifest.json") as f:
            manifest = json.load(f)

    except JSONDecodeError:
        raise ValueError(
            f"Manifest 文件不是合法 JSON：{manifest_path}"
        )

    if not isinstance(manifest, dict):
        raise ValueError(
            f"Manifest 根节点必须是 JSON 对象：{manifest_path}"
        )

    if "documents" not in manifest:
        raise ValueError(
            f"Manifest 缺少 documents 字段：{manifest_path}"
        )

    return manifest

def load_catalog():
    """
    加载目录文件
    """
    catalog_path = settings.CATALOG_PATH
    if not catalog_path.exists():
        raise FileNotFoundError(
            f"未找到手册版本目录文件：{catalog_path}"
        )
    else:
        with catalog_path.open("r", encoding="utf-8") as f:
            catalog = json.load(f)

    return catalog

def calculate_content_hash(file_path: Path):
    # 获取对应文件的sha256
    content = file_path.read_bytes()

    return hashlib.sha256(content).hexdigest()

def get_or_create_collection():
    """
    获取或创建chromadb
    """
    settings.EMBEDDING_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(settings.CHROMA_DIR),
    )

    return client.get_or_create_collection(
        name="inspection-knowledge",
    )

def reset_collection():
    client = chromadb.PersistentClient(
        path=str(settings.CHROMA_DIR)
    )

    try:
        client.delete_collection("inspection-knowledge")
    except ValueError:
        pass

    return client.get_or_create_collection(
        name="inspection-knowledge",
    )

def parse_document(file_path: str) -> dict:
    """
    读取 Markdown 文档，返回包含文档 ID、来源和正文的字典。
    文档解析结构包含：
    文档路径
    文档名
    文档正文
    文档类型
    文档引用

    """

    path = Path(file_path)

    # 1. 检查文件是否存在
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    # 2. 读取 Markdown 文件内容
    content = path.read_text(encoding="utf-8")

    # 3. 获取文件相对路径
    source_path = file_path.relative_to(BASE_DIR).as_posix()

    # 4. 获取文件名(有后缀&无后缀)
    document_id = path.stem
    document_name = path.name

    # 5. 获取该文件类型(是standards文档还是reports文档)
    if path.parent.name == "standards" or path.parent.name == "reports":
        document_type = path.parent.name
    else:
        raise ValueError(f"不支持的资料目录：{path.parent}")

    # 6. 通过正则表达查找来源
    if document_type == "reports":
        cited_ids = re.findall(
            r"\[来源：([^，,\]]+)",
            content,
        )
    else:
        cited_ids = []

    # 7. 获取文档的sha-256
    content_hash = calculate_content_hash(path)

    parsed_document = {
        "document_name": document_name,  # HX-QIM-PIPE-2025.md
        "document_id": document_id,    # HX-QIM-PIPE-2025
        "document_type": document_type,
        "source_path": source_path,
        "cited_ids": cited_ids,
        "content_hash": content_hash,
        "content": content,
    }

    return parsed_document

def split_markdown(document: dict) -> list[dict]:
    """
    将 Markdown 文档切分为可检索的知识块。

    先按二级、三级标题划分章节；当一个知识块超过字符上限时，
    再按固定长度切分，并在相邻块之间保留重叠正文。
    """
    max_chars = settings.RAG_CHUNK_MAX_CHARS
    overlap_chars = settings.RAG_CHUNK_OVERLAP_CHARS

    if max_chars <= 0:
        raise ValueError("RAG_CHUNK_MAX_CHARS 必须大于 0")

    if overlap_chars < 0:
        raise ValueError("RAG_CHUNK_OVERLAP_CHARS 不能小于 0")

    content = document["content"]
    lines = content.splitlines()
    chunks = []
    current_chapter = ""
    current_section = ""
    current_lines = []

    def append_current_chunk() -> None:
        """将当前章节正文切成一个或多个带上下文的知识块。"""
        body = "\n".join(current_lines).strip()

        if not body:
            return

        heading_lines = [current_chapter]
        if current_section:
            heading_lines.append(current_section)

        prefix = "\n".join(
            heading for heading in heading_lines if heading
        )
        separator = "\n\n" if prefix else ""
        body_limit = max_chars - len(prefix) - len(separator)

        if body_limit <= overlap_chars:
            raise ValueError(
                "RAG_CHUNK_MAX_CHARS 必须大于标题长度和 "
                "RAG_CHUNK_OVERLAP_CHARS 之和"
            )

        start = 0

        while start < len(body):
            end = min(start + body_limit, len(body))
            body_part = body[start:end]

            chunks.append(
                {
                    "chunk_id": (
                        f'{document["document_id"]}:chunk:{len(chunks)}'
                    ),
                    "document_id": document["document_id"],
                    "source_path": document["source_path"],
                    "content": f"{prefix}{separator}{body_part}",
                    "chapter_id": current_chapter,
                    "section_id": current_section,
                }
            )

            if end == len(body):
                break

            start = end - overlap_chars

    for line in lines:
        stripped_line = line.strip()

        if stripped_line.startswith("### "):
            # 新小节开始前，先写入上一个小节。
            append_current_chunk()
            current_section = stripped_line[4:]
            current_lines = []

        elif stripped_line.startswith("## "):
            # 新章节开始前，先写入上一个章节或小节。
            append_current_chunk()
            current_chapter = stripped_line[3:]
            current_section = ""
            current_lines = []

        elif stripped_line.startswith("# "):
            # 一级标题是整份文档标题，不作为独立知识块。
            continue

        else:
            # 仅记录已进入章节后的普通正文。
            if current_chapter:
                current_lines.append(line)

    # 最后一个章节或小节后没有新标题，因此循环结束后单独写入。
    append_current_chunk()

    return chunks

def is_citation_eligible(document: dict, catalog: dict) -> bool:
    """
    判断document中的引用是否可行
    """
    current_manuals = catalog["current_manuals"]

    if document["document_type"] == "standards":
        return document["document_id"] in current_manuals

    if document["document_type"] == "reports":
        cited_ids = document["cited_ids"]

        if not cited_ids:
            return False

        for cited_id in cited_ids:
            if cited_id not in current_manuals:
                return False

        return True

    raise ValueError(
        f'不支持的文档类型：{document["document_type"]}'
    )

def get_version_status(
        document: dict,
        catalog: dict,
) -> str:
    """返回文档的版本状态。"""

    if document["document_type"] == "standards":
        current_manuals = catalog["current_manuals"]

        if document["document_id"] in current_manuals:
            return "current"

        return "archived"

    if document["document_type"] == "reports":
        return "not_applicable"

    raise ValueError(
        f'不支持的文档类型：{document["document_type"]}'
    )

def build_change_plan(source_files: list[Path], manifest: dict, catalog: dict,) -> dict:
    """
    定义需要处理的文件
    """

    new_files = []
    changed_files = []
    unchanged_files = []
    eligibility_changed_files = []

    documents = manifest["documents"]
    current_source_paths = set()

    current_catalog_hash = calculate_content_hash(
        settings.CATALOG_PATH
    )

    catalog_changed = (
            manifest["catalog"]["content_sha256"] is not None
            and manifest["catalog"]["content_sha256"] != current_catalog_hash
    )

    for scope_path in source_files:
        source_path = scope_path.relative_to(BASE_DIR).as_posix()
        current_source_paths.add(source_path)

        current_hash = calculate_content_hash(scope_path)

        old_record = documents.get(source_path)

        if old_record is None:
            new_files.append(source_path)

        elif current_hash == old_record["content_sha256"]:
            # 文件正文没有变化；只有 catalog 变化时，才需要复核引用资格
            if catalog_changed:
                document = parse_document(scope_path)

                current_eligible = is_citation_eligible(
                    document,
                    catalog,
                )

                if current_eligible != old_record["citation_eligible"]:
                    eligibility_changed_files.append(source_path)
                else:
                    unchanged_files.append(source_path)

            else:
                unchanged_files.append(source_path)

        else:
            changed_files.append(source_path)

    deleted_files = []
    for source_path in documents:
        if source_path not in current_source_paths:
            deleted_files.append(source_path)

    current_embedding = get_embedding_config()
    has_existing_index = bool(
        manifest["documents"]
    )
    embedding_changed = (
            has_existing_index
            and manifest["embedding"] != current_embedding
    )


    return {
        "all_source_files": [
            path.relative_to(BASE_DIR).as_posix()
            for path in source_files
        ],
        "new_files": new_files,
        "changed_files": changed_files,
        "unchanged_files": unchanged_files,
        "deleted_files": deleted_files,
        "eligibility_changed_files": eligibility_changed_files,
        "catalog_changed": catalog_changed,
        "embedding_changed": embedding_changed,
        "requires_rebuild": embedding_changed,
    }

def save_manifest(manifest: dict) -> None:
    """将最新索引记录保存到本地 manifest.json。"""

    manifest_dir = settings.INDEX_MANIFEST_DIR
    manifest_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / "manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            manifest,
            file,
            ensure_ascii=False,
            indent=2,
        )

def apply_change_plan(
        change_plan: dict,
        manifest: dict,
        catalog: dict,
) -> dict:
    """
    根据change_plan更新manifest与catalog，以及向量化并入库
    """

    if change_plan["requires_rebuild"]:
        collection = reset_collection()

        # 旧 manifest 中的 chunk_ids 已经无效
        manifest = create_empty_manifest()

        files_to_index = change_plan["all_source_files"]

    else:
        collection = get_or_create_collection()

        files_to_index = (
                change_plan["new_files"]
                + change_plan["changed_files"]
        )

        files_to_remove = (
                change_plan["changed_files"]
                + change_plan["deleted_files"]
        )
        for source_path in files_to_remove:
            old_record = manifest["documents"][source_path]

            if old_record["chunk_ids"]:
                collection.delete(
                    ids=old_record["chunk_ids"]
                )

            # 源文件已经彻底删除，才从 manifest 中移除记录
            if source_path in change_plan["deleted_files"]:
                del manifest["documents"][source_path]

    for source_path in files_to_index:
        file_path = BASE_DIR / source_path

        document = parse_document(file_path)

        cit_eligible = is_citation_eligible(document, catalog)
        version_status = get_version_status(document, catalog)

        manifest["documents"][source_path] = {
            "document_id": document["document_id"],
            "document_type": document["document_type"],
            "version_status": version_status,
            "citation_eligible": cit_eligible,
            "content_sha256": document["content_hash"],
            "chunk_ids": [],
        }

        if not cit_eligible:
            continue

        chunks = split_markdown(document)

        texts = [chunk["content"] for chunk in chunks]
        embeddings = create_embeddings(texts)
        chunk_ids = [
            chunk["chunk_id"]
            for chunk in chunks
        ]

        collection.upsert(
            ids=chunk_ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=[
                {
                    "document_id": chunk["document_id"],
                    "source_path": chunk["source_path"],
                    "chapter_id": chunk["chapter_id"],
                    "section_id": chunk["section_id"],
                    "version_status": version_status,
                    "citation_eligible": cit_eligible,
                }
                for chunk in chunks
            ],
        )

        manifest["documents"][source_path]["chunk_ids"] = chunk_ids


    return manifest

async def sync_knowledge_base(
        scope: str = "all",
        mode: str = "dry_run"
):
    # 获得知识库文件路径
    scope_files = await collect_source_files(scope)
    # 获得向量库清单
    manifest = load_manifest()
    catalog = load_catalog()

    change_plan = build_change_plan(scope_files, manifest, catalog)

    if change_plan["requires_rebuild"]:
        rebuild_reason = "embedding 模型配置已变化"
    else:
        rebuild_reason = ""

    if mode == "dry_run":


        return {
            "success": True,
            "mode": "dry_run",
            **change_plan,
            "indexed_files": [],
            "rebuild_reason": rebuild_reason,
            "message": "变更预览完成，未写入向量库"
        }

    # if change_plan["changed_files"]:
    #     raise NotImplementedError(
    #         "当前版本暂未实现变更文件的重新入库。"
    #     )
    #
    # if change_plan["deleted_files"]:
    #     raise NotImplementedError(
    #         "当前版本暂未实现删除文件对应的向量。"
    #     )

    updated_manifest = apply_change_plan(
        change_plan,
        manifest,
        catalog,
    )

    save_manifest(updated_manifest)

    return {
        "success": True,
        "mode": "apply",
        **change_plan,
        "indexed_files": change_plan["new_files"],
        "rebuild_reason": rebuild_reason,
        "message": "已完成向量库更新。",
    }

if __name__ == "__main__":
    # import asyncio
    #
    # asyncio.run(sync_knowledge_base())
    # asyncio.run(collect_source_files())

    pipe_document = parse_document(
        BASE_DIR / "data" / "standards" / "HX-QIM-PIPE-2025.md"
    )

    report_document = parse_document(
        BASE_DIR / "data" / "reports" / "HX-NDT-2025-041.md"
    )

    print(report_document["document_id"])
    print(report_document["document_type"])
    print(report_document["source_path"])
    print(report_document["cited_ids"])
    print(report_document["content_hash"])
