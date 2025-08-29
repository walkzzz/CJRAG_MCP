#!/usr/bin/env python3
"""
SQLite + sqlite-vec 完整示例
1. 递归扫描 docs 目录下 *.md
2. 切分 chunk
3. 向量化 (768 维)
4. 存入 md_vec.db
5. CLI 交互查询
"""

import sqlite3, pathlib, json, argparse
import sqlite_vec               # 0.1.1 及以上
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import MarkdownTextSplitter

DB_PATH   = pathlib.Path(__file__).with_name("md_vec.db")
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
def ingest(root: pathlib.Path):
    encoder  = SentenceTransformer("all-MiniLM-L6-v2")
    splitter = MarkdownTextSplitter(chunk_size=CHUNK_SZ, chunk_overlap=OVERLAP)

    conn = sqlite3.connect(DB_PATH)
    try:
        init_db(conn)
        cur = conn.cursor()
        for md in root.rglob("*.md"):
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
def query(q: str, k: int = 5):
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vec_q   = encoder.encode([q]).astype("float32").tobytes()

    conn = sqlite3.connect(DB_PATH)
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
def build_vec_db(root: str | pathlib.Path = "docs"):
    """
    把指定目录下的所有 markdown 文件向量化后存入 SQLite。
    root: markdown 根目录
    """
    ingest(pathlib.Path(root).expanduser().resolve())

def search_vec_db(query_text: str, top_k: int = 5) -> list[dict]:
    """
    在向量数据库中做相似度搜索。
    返回 [{'doc':..., 'chunk':..., 'score':...}, ...]
    """
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    vec_q = encoder.encode([query_text]).astype("float32").tobytes()

    conn = sqlite3.connect(DB_PATH)
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

def clear_vec_db():
    """
    清空数据库（chunks 与向量表）
    """
    conn = sqlite3.connect(DB_PATH)
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
    build_vec_db("docs")

    # # 2. 查询
    # results = search_vec_db("如何配置 CUDA", top_k=3)
    # for r in results:
    #     print(r["score"], r["chunk"][:100])

    # # 3. 清空
    # clear_vec_db()

    # ap = argparse.ArgumentParser()
    # ap.add_argument("--docs", type=pathlib.Path, default="docs",
    #                 help="markdown 根目录")
    # ap.add_argument("--ingest", action="store_true",
    #                 help="执行入库")
    # ap.add_argument("--query", type=str,
    #                 help="查询文本")
    # args = ap.parse_args()

    # if args.ingest:
    #     ingest(args.docs.expanduser().resolve())
    # elif args.query:
    #     query(args.query)
    # else:
    #     ap.print_help()
