#!/usr/bin/env python3
"""
单元测试文件，用于测试 md_sqlite3.py 中的功能。
"""

import unittest
import tempfile
import os
import pathlib
import sqlite3
import time
import numpy as np
from unittest.mock import patch, MagicMock

# 假设 main.py 与 test_main.py 在同一目录下
import md_sqlite3

class TestMainFunctions(unittest.TestCase):
    
    def setUp(self):
        """在每个测试方法之前执行，创建临时数据库文件。"""
        # 创建临时目录和文件
        self.test_dir = tempfile.TemporaryDirectory()
        self.temp_md_file = pathlib.Path(self.test_dir.name) / "test.md"
        self.temp_db_path = md_sqlite3.get_db_path(str(self.temp_md_file))
        self.conn = None  # 用于存储数据库连接
    
    def tearDown(self):
        """在每个测试方法之后执行，清理临时文件。"""
        # 确保所有数据库连接都已关闭
        if self.conn:
            self.conn.close()
        # 等待一段时间，确保所有文件句柄都被释放
        time.sleep(0.1)
        # 清理临时目录
        self.test_dir.cleanup()
    
    def test_init_db(self):
        """测试 init_db 函数是否能正确初始化数据库。"""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as tmp_db:
            tmp_db_path = tmp_db.name
        
        try:
            self.conn = sqlite3.connect(tmp_db_path)
            md_sqlite3.init_db(self.conn)
            # 检查表是否创建成功
            cur = self.conn.cursor()
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='chunks'")
            self.assertIsNotNone(cur.fetchone())
            cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='vec_chunks'")
            self.assertIsNotNone(cur.fetchone())
            self.conn.close()
            self.conn = None
        finally:
            os.unlink(tmp_db_path)
    
    @patch('md_sqlite3.SentenceTransformer')
    @patch('md_sqlite3.MarkdownTextSplitter')
    def test_ingest(self, mock_splitter, mock_sentence_transformer):
        # 模拟 SentenceTransformer
        mock_encoder_instance = MagicMock()
        mock_sentence_transformer.return_value = mock_encoder_instance
        # 模拟 SentenceTransformer 的 encode 方法返回 384 维向量
        mock_vector = np.random.rand(1, 384).astype(np.float32)
        mock_encoder_instance.encode.return_value = mock_vector
        
        # 模拟 MarkdownTextSplitter
        mock_splitter_instance = MagicMock()
        mock_splitter.return_value = mock_splitter_instance
        mock_splitter_instance.split_text.return_value = ["chunk1", "chunk2"]

        # 创建一个临时的 markdown 文件
        test_md_content = "# Test Heading\n\nThis is a test markdown file."
        test_md_path = self.temp_md_file
        with open(test_md_path, 'w', encoding='utf-8') as f:
            f.write(test_md_content)

        # 调用 ingest 函数
        md_sqlite3.ingest(pathlib.Path(self.test_dir.name))

        # 验证数据库中是否有数据插入
        self.conn = sqlite3.connect(self.temp_db_path)
        cur = self.conn.execute("SELECT COUNT(*) FROM chunks")
        count = cur.fetchone()[0]
        self.conn.close()
        self.conn = None
        self.assertEqual(count, 2)  # 应该有2个chunk被插入
    
    def test_clear_vec_db(self):
        """测试 clear_vec_db 函数是否能正确清空数据库。"""
        # 先插入一些数据
        self.conn = sqlite3.connect(self.temp_db_path)
        md_sqlite3.init_db(self.conn)
        self.conn.execute("INSERT INTO chunks(doc, chunk) VALUES (?, ?)", ("test_doc", "test_chunk"))
        rowid = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        # 插入一个有效的 384 维向量
        valid_vector = np.random.rand(384).astype(np.float32).tobytes()
        self.conn.execute("INSERT INTO vec_chunks(rowid, embedding) VALUES (?, ?)", (rowid, valid_vector))
        self.conn.commit()
        self.conn.close()
        self.conn = None
        
        # 调用 clear_vec_db
        md_sqlite3.clear_vec_db(str(self.temp_md_file))
        
        # 验证数据是否被清空
        self.conn = sqlite3.connect(self.temp_db_path)
        md_sqlite3.init_db(self.conn)  # 确保 sqlite-vec 扩展被加载
        cur = self.conn.execute("SELECT COUNT(*) FROM chunks")
        count_chunks = cur.fetchone()[0]
        cur = self.conn.execute("SELECT COUNT(*) FROM vec_chunks")
        count_vec_chunks = cur.fetchone()[0]
        self.conn.close()
        self.conn = None
        self.assertEqual(count_chunks, 0)
        self.assertEqual(count_vec_chunks, 0)
    
    def test_query(self):
        """测试 query 函数是否能正确执行。"""
        # 创建一个临时的 markdown 文件
        test_md_content = "# Test Heading\n\nThis is a test markdown file with some content for querying."
        test_md_path = self.temp_md_file
        with open(test_md_path, 'w', encoding='utf-8') as f:
            f.write(test_md_content)
        
        # 调用 ingest 函数，将文件内容插入数据库
        md_sqlite3.ingest(pathlib.Path(self.test_dir.name))
        
        # 调用 query 函数，验证函数是否能正常执行而不抛出异常
        query_text = "test"
        try:
            md_sqlite3.query(query_text, str(self.temp_md_file))
            # 如果没有抛出异常，则测试通过
            self.assertTrue(True)
        except Exception as e:
            # 如果抛出异常，则测试失败
            self.fail(f"query 函数执行时抛出异常: {e}")
    
    def test_search_vec_db(self):
        """测试 search_vec_db 函数是否能正确执行。"""
        # 创建一个临时的 markdown 文件
        test_md_content = "# Test Heading\n\nThis is a test markdown file with some content for searching."
        test_md_path = self.temp_md_file
        with open(test_md_path, 'w', encoding='utf-8') as f:
            f.write(test_md_content)
        
        # 调用 ingest 函数，将文件内容插入数据库
        md_sqlite3.ingest(pathlib.Path(self.test_dir.name))
        
        # 调用 search_vec_db 函数，验证函数是否能正常执行而不抛出异常
        query_text = "test"
        try:
            results = md_sqlite3.search_vec_db(query_text, str(self.temp_md_file), top_k=5)
            # 如果没有抛出异常，则测试通过
            self.assertIsInstance(results, list)
        except Exception as e:
            # 如果抛出异常，则测试失败
            self.fail(f"search_vec_db 函数执行时抛出异常: {e}")

if __name__ == '__md_sqlite3__':
    unittest.main()
