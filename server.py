from mcp.server.fastmcp import FastMCP
from typing import List, Tuple, Union, Optional, Iterable
from md_sqlite3 import *
# 注册 MCP 服务，用于通过 MCP 协议接收绘图命令
mcp = FastMCP("CJRAGServer")

# ------------------------------
# 绘图工具函数(2D)
# ------------------------------
@mcp.tool()
def build_vec_db(root: str | pathlib.Path = "docs", save_path: str | None = None):
    """
    把指定目录下的所有 markdown 文件向量化后存入 SQLite。
    root: markdown 根目录
    save_path: 数据库存储路径
    """
    ingest(pathlib.Path(root).expanduser().resolve(), save_path)
@mcp.tool()
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
@mcp.tool()
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
# ------------------------------------------
# 主函数入口
# ------------------------------------------
if __name__ == "__main__":
    mcp.run(transport="stdio")
