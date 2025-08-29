#!/usr/bin/env python3
"""
SQLite + sqlite-vec 完整示例
1. 递归扫描 docs 目录下 *.md
2. 切分 chunk
3. 向量化 (384 维)
4. 存入 {md_file_name}_vec.db
5. CLI 交互查询
"""

import sqlite3, pathlib, json, argparse
import sqlite_vec               # 0.1.1 及以上
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownTextSplitter

def get_db_path(md_file_path: str, save_path: str | None = None) -> pathlib.Path:
    """
    根据.md文件的名称生成数据库文件路径
    """
    md_file_name = pathlib.Path(md_file_path).stem
    if save_path:
        return pathlib.Path(save_path) / f"{md_file_name}_vec.db"
    # 如果没有提供保存路径，则使用md文件所在的路径
    return pathlib.Path(md_file_path).parent / f"{md_file_name}_vec.db"

VEC_DIM   = 384                 # all-MiniLM-L6-v2 输出维度
CHUNK_SZ  = 256
OVERLAP   = 30

# -------------------------------------------------
# 建库 & 建表
# -------------------------------------------------
def init_db(conn: sqlite3.Connection):
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)      # 加载 sqlite-vec 扩展
    conn.executescript("""
    CREATE TABLE IF NOT EXISTS chunks(
        id      INTEGER PRIMARY KEY,
        doc     TEXT,
        heading TEXT,
        chunk   TEXT
    );

    -- 向量虚拟表：rowid 与 chunks.id 一一对应
    CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
        embedding float[384]
    );
    """)

# -------------------------------------------------
# 解析 + 向量化 + 入库
# -------------------------------------------------
def ingest(root: pathlib.Path, save_path: str | None = None):
    encoder  = SentenceTransformer("all-MiniLM-L6-v2")
    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SZ, chunk_overlap=OVERLAP)

    for md in root.rglob("*.md"):
        # 为每个.md文件创建一个单独的数据库
        db_path = get_db_path(str(md), save_path)
        if save_path:
            db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        try:
            init_db(conn)
            cur = conn.cursor()
            text = md.read_text(encoding="utf-8")
            for ch in splitter.split_text(text):
                cur.execute(
                    "INSERT INTO chunks(doc, chunk) VALUES (?, ?)",
                    (str(md), ch)
                )
                rowid = cur.lastrowid
                vec = encoder.encode([ch]).astype("float32").tobytes()
                cur.execute(
                    "INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)",
                    (rowid, vec)
                )
            conn.commit()
        finally:
            conn.close()
    print("ingest done")

# -------------------------------------------------
# 向量查询
# -------------------------------------------------
def query(q: str, md_file_path: str, k: int = 5, save_path: str | None = None):
    """
    在指定的.md文件对应的向量数据库中进行查询
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vec_q   = encoder.encode([q]).astype("float32").tobytes()

    # 根据.md文件路径获取对应的数据库路径
    db_path = get_db_path(md_file_path, save_path)
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        cur = conn.execute("""
            SELECT chunks.doc, chunks.chunk,
                   vec_distance_cosine(vec_chunks.embedding, ?) AS score
            FROM vec_chunks
            JOIN chunks ON chunks.id = vec_chunks.rowid
            ORDER BY score
            LIMIT ?
        """, (vec_q, k))
        for doc, chunk, score in cur.fetchall():
            print(f"[{score:.4f}] {doc}")
            print(chunk[:200] + "...\n")
    finally:
        conn.close()

# -------------------------------------------------
# 工具函数（取代原来的 __main__）
# -------------------------------------------------
def build_vec_db(root: str | pathlib.Path = "docs", save_path: str | None = None):
    """
    把指定目录下的所有 markdown 文件向量化后存入 SQLite。
    root: markdown 根目录
    save_path: 数据库存储路径
    """
    ingest(pathlib.Path(root).expanduser().resolve(), save_path)

def search_vec_db(query_text: str, md_file_path: str, top_k: int = 5, save_path: str | None = None) -> list[dict]:
    """
    在指定的.md文件对应的向量数据库中做相似度搜索。
    返回 [{'doc':..., 'chunk':..., 'score':...}, ...]
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vec_q = encoder.encode([query_text]).astype("float32").tobytes()
    
    # 根据.md文件路径获取对应的数据库路径
    db_path = get_db_path(md_file_path, save_path)
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)
        cur = conn.execute("""
            SELECT chunks.doc      AS doc,
                   chunks.chunk    AS chunk,
                   vec_distance_cosine(vec_chunks.embedding, ?) AS score
            FROM vec_chunks
            JOIN chunks ON chunks.id = vec_chunks.rowid
            ORDER BY score
            LIMIT ?
        """, (vec_q, top_k))
        rows = cur.fetchall()
    finally:
        conn.close()
    return [{"doc": r[0], "chunk": r[1], "score": r[2]} for r in rows]

def clear_vec_db(md_file_path: str, save_path: str | None = None):
    """
    清空指定.md文件对应的数据库（chunks 与向量表）
    """
    # 根据.md文件路径获取对应的数据库路径
    db_path = get_db_path(md_file_path, save_path)
    conn = sqlite3.connect(db_path)
    try:
        init_db(conn)  # 确保 sqlite-vec 扩展被加载
        conn.execute("DELETE FROM chunks")
        # 删除并重新创建 vec_chunks 虚拟表以清空数据
        conn.execute("DROP TABLE vec_chunks")
        conn.execute("""
        CREATE VIRTUAL TABLE vec_chunks USING vec0(
            embedding float[384]
        );
        """)
        conn.commit()
    finally:
        conn.close()




# -------------------------------------------------
# CLI
# -------------------------------------------------
if __name__ == "__main__":
    # 1. 构建 / 更新向量库
    # build_vec_db("./")

    # 2. 查询
    # results = search_vec_db("eDSL", top_k=3)
    # for r in results:
    #     print(r["score"], r["chunk"][:100])

    # # 3. 清空
    # clear_vec_db()

    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", type=pathlib.Path, default="docs",
                    help="markdown 根目录")
    ap.add_argument("--ingest", action="store_true",
                    help="执行入库")
    ap.add_argument("--query", type=str,
                    help="查询文本")
    ap.add_argument("--md-file", type=str,
                    help="指定要查询或清空的.md文件路径")
    ap.add_argument("--save-path", type=str,
                    help="数据库保存路径")
    args = ap.parse_args()

    if args.ingest:
        ingest(args.docs.expanduser().resolve(), args.save_path)
    elif args.query and args.md_file:
        results = search_vec_db(args.query, args.md_file, top_k=5, save_path=args.save_path)
        for r in results:
            print(f"[{r['score']:.4f}] {r['doc']}")
            print(r["chunk"][:200] + "...\n")
    elif args.query:
        print("请提供 --md-file 参数以指定要查询的.md文件")
    else:
        ap.print_help()
