"""
SQL 标识符验证测试

测试 src/data/base.py 的 validate_sql_identifier 函数，包括：
- 有效标识符验证
- 无效标识符拒绝
- SQL 注入防护
- 中文字符支持
"""

import pytest

from src.data.base import validate_sql_identifier


class TestValidateSqlIdentifier:
    """SQL 标识符验证测试"""

    def test_valid_simple_name(self):
        """测试简单名称"""
        assert validate_sql_identifier("users", "表名") == "users"
        assert validate_sql_identifier("my_table", "表名") == "my_table"
        assert validate_sql_identifier("Table1", "表名") == "Table1"

    def test_valid_underscore_prefix(self):
        """测试下划线开头的名称"""
        assert validate_sql_identifier("_private", "列名") == "_private"
        assert validate_sql_identifier("__internal", "列名") == "__internal"

    def test_valid_chinese_name(self):
        """测试中文名称"""
        assert validate_sql_identifier("用户表", "表名") == "用户表"
        assert validate_sql_identifier("订单_明细", "表名") == "订单_明细"
        assert validate_sql_identifier("field字段", "列名") == "field字段"

    def test_valid_numbers_in_name(self):
        """测试包含数字的名称"""
        assert validate_sql_identifier("table1", "表名") == "table1"
        assert validate_sql_identifier("v2_users", "表名") == "v2_users"

    def test_invalid_empty_name(self):
        """测试空名称"""
        with pytest.raises(ValueError, match="表名不能为空"):
            validate_sql_identifier("", "表名")

        with pytest.raises(ValueError, match="表名不能为空"):
            validate_sql_identifier(None, "表名")

    def test_invalid_non_string(self):
        """测试非字符串类型"""
        with pytest.raises(ValueError, match="必须是字符串类型"):
            validate_sql_identifier(123, "表名")

        with pytest.raises(ValueError, match="必须是字符串类型"):
            validate_sql_identifier(["table"], "表名")

    def test_invalid_too_long(self):
        """测试过长的名称"""
        long_name = "a" * 129
        with pytest.raises(ValueError, match="长度不能超过 128 字符"):
            validate_sql_identifier(long_name, "表名")

    def test_invalid_special_characters(self):
        """测试特殊字符"""
        invalid_names = [
            "table;drop",  # 分号
            "table--comment",  # SQL 注释
            "table/**/",  # 块注释
            "table'quote",  # 单引号
            'table"double',  # 双引号
            "table`backtick",  # 反引号
            "table name",  # 空格
            "table\ttab",  # 制表符
            "table\nnewline",  # 换行符
            "table$dollar",  # 美元符
            "table@at",  # @符号
            "table#hash",  # 井号
            "table%percent",  # 百分号
            "table(paren",  # 括号
            "table=equal",  # 等号
            "table.dot",  # 点号
        ]

        for name in invalid_names:
            with pytest.raises(ValueError, match="包含非法字符"):
                validate_sql_identifier(name, "表名")

    def test_sql_injection_attempts(self):
        """测试 SQL 注入尝试"""
        injection_attempts = [
            "users; DROP TABLE users;--",
            "users' OR '1'='1",
            "users`; DELETE FROM users;`",
            "users UNION SELECT * FROM passwords",
            "1 OR 1=1",
        ]

        for attempt in injection_attempts:
            with pytest.raises(ValueError, match="包含非法字符"):
                validate_sql_identifier(attempt, "表名")

    def test_max_length_boundary(self):
        """测试边界长度"""
        # 128 字符应该可以
        name_128 = "a" * 128
        assert validate_sql_identifier(name_128, "表名") == name_128

        # 129 字符应该失败
        name_129 = "a" * 129
        with pytest.raises(ValueError, match="长度不能超过 128 字符"):
            validate_sql_identifier(name_129, "表名")


class TestSqlInjectionPrevention:
    """SQL 注入防护集成测试"""

    def test_sqlite_pool_rejects_invalid_table_name(self, tmp_path):
        """测试 SQLiteTaskPool 拒绝无效表名"""
        import sqlite3
        from src.data.sqlite import SQLiteTaskPool

        # 创建测试数据库
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE users (
                id INTEGER PRIMARY KEY,
                name TEXT,
                result TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        # 尝试使用危险的表名
        with pytest.raises(ValueError, match="包含非法字符"):
            SQLiteTaskPool(
                db_path=str(db_path),
                table_name="users; DROP TABLE users;--",
                columns_to_extract=["name"],
                columns_to_write={"answer": "result"},
            )

    def test_sqlite_pool_rejects_invalid_column_name(self, tmp_path):
        """测试 SQLiteTaskPool 拒绝无效列名"""
        import sqlite3
        from src.data.sqlite import SQLiteTaskPool

        # 创建测试数据库
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE tasks (
                id INTEGER PRIMARY KEY,
                input TEXT,
                output TEXT
            )
        """
        )
        conn.commit()
        conn.close()

        # 尝试使用危险的列名
        with pytest.raises(ValueError, match="包含非法字符"):
            SQLiteTaskPool(
                db_path=str(db_path),
                table_name="tasks",
                columns_to_extract=["input; DROP TABLE tasks;--"],
                columns_to_write={"answer": "output"},
            )
