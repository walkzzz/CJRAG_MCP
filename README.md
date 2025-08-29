# CJRAG_MCP

CJRAG_MCP 是一个基于 SQLite 和 sqlite-vec 的向量检索增强生成（RAG）系统。它能够将 Markdown 文档向量化并存储在 SQLite 数据库中，然后通过语义搜索来检索相关内容。

## 功能特性

- 递归扫描指定目录下的 Markdown 文件
- 使用 `all-MiniLM-L6-v2` 模型对文档进行向量化（384维）
- 将向量数据存储在 SQLite 数据库中
- 支持通过命令行接口（CLI）进行语义搜索
- 提供 MCP 工具函数，可集成到其他系统中

## 依赖

- Python 3.13+
- sqlite-vec
- sentence-transformers
- langchain-text-splitters
- mcp

## 安装

```bash
pip install cjrag-mcp
```

或者使用 uv 初始化项目：

```bash
uv init cjrag-mcp
cd cjrag-mcp
uv sync
```

## 使用方法

### 命令行使用

1. 将指定目录下的 Markdown 文件向量化并存储到 SQLite 数据库：

```bash
python md_sqlite3.py --docs <docs_directory> --ingest [--save-path <save_directory>]
```

也可以直接运行脚本构建向量库：

```bash
python md_sqlite3.py
```

2. 查询相关内容：

```bash
python md_sqlite3.py --query <query_text> --md-file <md_file_path> [--save-path <save_directory>]
```

3. 直接运行脚本进行查询：

```bash
python md_sqlite3.py
```

注意：当使用 `--save-path` 参数时，应指定目录路径而非文件路径。

### MCP 工具函数

项目还提供了 MCP 工具函数，可以在其他系统中集成使用：

- `build_vec_db(root, save_path)`: 构建向量数据库
- `search_vec_db(query_text, md_file_path, top_k, save_path)`: 搜索相关内容
- `clear_vec_db(md_file_path, save_path)`: 清空数据库

## 项目结构

- `md_sqlite3.py`: 主要的向量处理和搜索功能
- `server.py`: MCP 服务器实现
- `tutorial.md`: 使用教程

## 许可证

[MIT](LICENSE)