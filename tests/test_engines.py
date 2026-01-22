"""
数据引擎测试

测试 src/data/engines/ 的引擎实现
"""

import pytest
import pandas as pd


class TestEngineFactory:
    """引擎工厂测试"""

    def test_get_pandas_engine(self):
        """测试获取 Pandas 引擎"""
        from src.data.engines import get_engine

        engine = get_engine("pandas")
        assert engine.name == "pandas"

    def test_get_auto_engine(self):
        """测试自动选择引擎"""
        from src.data.engines import get_engine, POLARS_AVAILABLE

        engine = get_engine("auto")

        if POLARS_AVAILABLE:
            assert engine.name == "polars"
        else:
            assert engine.name == "pandas"

    def test_get_available_libraries(self):
        """测试获取可用库状态"""
        from src.data.engines import get_available_libraries

        libs = get_available_libraries()

        # 核心库必须可用
        assert libs["pandas"] is True
        assert libs["openpyxl"] is True

        # 返回所有库状态
        assert "numpy" in libs
        assert "polars" in libs
        assert "fastexcel" in libs
        assert "xlsxwriter" in libs

    def test_reader_writer_auto(self):
        """测试读写器自动选择"""
        from src.data.engines import (
            get_engine,
            FASTEXCEL_AVAILABLE,
            XLSXWRITER_AVAILABLE,
        )

        engine = get_engine("pandas", "auto", "auto")

        if FASTEXCEL_AVAILABLE:
            assert engine.excel_reader == "calamine"
        else:
            assert engine.excel_reader == "openpyxl"

        if XLSXWRITER_AVAILABLE:
            assert engine.excel_writer == "xlsxwriter"
        else:
            assert engine.excel_writer == "openpyxl"

    def test_fallback_on_unavailable(self):
        """测试不可用时回退"""
        from src.data.engines import get_engine

        # 即使请求 polars，不可用时应回退到 pandas
        engine = get_engine("polars")
        assert engine.name in ["pandas", "polars"]


class TestPandasEngine:
    """PandasEngine 测试"""

    def test_read_excel(self, pandas_engine, sample_excel_file):
        """测试读取 Excel"""
        df = pandas_engine.read_excel(sample_excel_file)

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert "id" in df.columns
        assert "question" in df.columns

    def test_write_excel(self, pandas_engine, sample_dataframe, tmp_path):
        """测试写入 Excel"""
        output_path = tmp_path / "output.xlsx"

        pandas_engine.write_excel(sample_dataframe, output_path)

        assert output_path.exists()

        # 验证写入内容
        df_read = pandas_engine.read_excel(output_path)
        assert len(df_read) == len(sample_dataframe)

    def test_get_row(self, pandas_engine, sample_dataframe):
        """测试获取行"""
        row = pandas_engine.get_row(sample_dataframe, 0)

        assert isinstance(row, dict)
        assert row["id"] == 1
        assert row["question"] == "Q1"

    def test_get_row_invalid_index(self, pandas_engine, sample_dataframe):
        """测试无效索引"""
        with pytest.raises(IndexError):
            pandas_engine.get_row(sample_dataframe, 999)

    def test_set_value(self, pandas_engine, sample_dataframe):
        """测试设置值"""
        df = sample_dataframe.copy()
        pandas_engine.set_value(df, 0, "answer", "New Answer")

        assert df.at[0, "answer"] == "New Answer"

    def test_is_empty(self, pandas_engine):
        """测试空值判断"""
        assert pandas_engine.is_empty(None) is True
        assert pandas_engine.is_empty("") is True
        assert pandas_engine.is_empty("   ") is True
        assert pandas_engine.is_empty("text") is False
        assert pandas_engine.is_empty(0) is False

    def test_is_empty_vectorized(self, pandas_engine, sample_dataframe):
        """测试向量化空值判断"""
        result = pandas_engine.is_empty_vectorized(sample_dataframe["question"])

        import pandas as pd

        assert isinstance(result, pd.Series)
        # 索引 2 是 ""，索引 3 是 None
        assert result.iloc[2]  # "" 是空
        assert result.iloc[3]  # None 是空
        assert not result.iloc[0]  # "Q1" 非空

    def test_is_empty_vectorized_string_dtype(self, pandas_engine):
        """测试 string dtype 的空值判断"""
        series = pd.Series(["", "text", None], dtype="string")
        result = pandas_engine.is_empty_vectorized(series)

        assert result.tolist() == [True, False, True]

    def test_filter_indices(self, pandas_engine, sample_dataframe):
        """测试条件过滤"""
        # 过滤非空的 question
        indices = pandas_engine.filter_indices(
            sample_dataframe, "question", "not_empty"
        )

        assert 0 in indices  # Q1
        assert 1 in indices  # Q2
        assert 4 in indices  # Q5
        assert 2 not in indices  # ""
        assert 3 not in indices  # None

    def test_filter_indices_vectorized(self, pandas_engine, sample_dataframe):
        """测试向量化过滤未处理行"""
        indices = pandas_engine.filter_indices_vectorized(
            sample_dataframe,
            input_columns=["question"],
            output_columns=["answer"],
            require_all_inputs=True,
        )

        # question 非空且 answer 为空的行
        assert 0 in indices  # Q1, answer=None
        assert 4 in indices  # Q5, answer=None
        assert 1 not in indices  # Q2, answer=A2 (已有答案)

    def test_row_count(self, pandas_engine, sample_dataframe):
        """测试行数统计"""
        assert pandas_engine.row_count(sample_dataframe) == 5

    def test_get_column_names(self, pandas_engine, sample_dataframe):
        """测试获取列名"""
        columns = pandas_engine.get_column_names(sample_dataframe)

        assert "id" in columns
        assert "question" in columns
        assert "context" in columns
        assert "answer" in columns

    def test_has_column(self, pandas_engine, sample_dataframe):
        """测试列存在检查"""
        assert pandas_engine.has_column(sample_dataframe, "question") is True
        assert pandas_engine.has_column(sample_dataframe, "nonexistent") is False

    def test_add_column(self, pandas_engine, sample_dataframe):
        """测试添加列"""
        df = pandas_engine.add_column(sample_dataframe.copy(), "new_col", "default")

        assert "new_col" in df.columns
        assert df["new_col"].iloc[0] == "default"

    def test_copy(self, pandas_engine, sample_dataframe):
        """测试复制 DataFrame"""
        df_copy = pandas_engine.copy(sample_dataframe)

        # 修改副本不影响原始
        df_copy.at[0, "question"] = "Modified"
        assert sample_dataframe.at[0, "question"] == "Q1"


class TestPolarsEngine:
    """PolarsEngine 测试"""

    @pytest.fixture(autouse=True)
    def skip_if_polars_unavailable(self):
        """如果 Polars 不可用则跳过整个测试类"""
        from src.data.engines import POLARS_AVAILABLE

        if not POLARS_AVAILABLE:
            pytest.skip("Polars not available on this platform")

    def test_read_excel(self, polars_engine, sample_excel_file):
        """测试读取 Excel"""
        import polars as pl

        df = polars_engine.read_excel(sample_excel_file)

        assert isinstance(df, pl.DataFrame)
        assert len(df) == 5

    def test_row_count(self, polars_engine, sample_excel_file):
        """测试行数统计"""
        df = polars_engine.read_excel(sample_excel_file)
        assert polars_engine.row_count(df) == 5

    def test_get_column_names(self, polars_engine, sample_excel_file):
        """测试获取列名"""
        df = polars_engine.read_excel(sample_excel_file)
        columns = polars_engine.get_column_names(df)

        assert "id" in columns
        assert "question" in columns

    def test_is_empty(self, polars_engine):
        """测试空值判断"""
        assert polars_engine.is_empty(None) is True
        assert polars_engine.is_empty("") is True
        assert polars_engine.is_empty("text") is False

    def test_set_value(self, polars_engine):
        """测试设置值并返回新 DataFrame"""
        import polars as pl

        df = pl.DataFrame({"input": ["a", "b"], "output": ["", "done"]})
        updated = polars_engine.set_value(df, 0, "output", "x")

        assert updated is not df
        assert polars_engine.get_row(updated, 0)["output"] == "x"
        assert polars_engine.get_row(df, 0)["output"] == ""

    def test_set_values_batch(self, polars_engine):
        """测试批量更新"""
        import polars as pl

        df = pl.DataFrame({"output": ["", "", "done"]})
        updates = [(0, "output", "v0"), (1, "output", "v1")]
        updated = polars_engine.set_values_batch(df, updates)

        assert polars_engine.get_row(updated, 0)["output"] == "v0"
        assert polars_engine.get_row(updated, 1)["output"] == "v1"

    def test_filter_indices_vectorized_with_offset(self, polars_engine):
        """测试分片过滤返回全局索引"""
        import polars as pl

        df = pl.DataFrame(
            {"input": ["a"] * 10, "output": ["done"] * 5 + [""] * 3 + ["done"] * 2}
        )

        sub_df = polars_engine.slice_by_index_range(df, 3, 8)
        indices = polars_engine.filter_indices_vectorized(
            sub_df,
            input_columns=["input"],
            output_columns=["output"],
            require_all_inputs=True,
            index_offset=3,
        )

        assert indices == [5, 6, 7]

    def test_write_excel(self, polars_engine, tmp_path):
        """测试写回 Excel"""
        from src.data.engines import XLSXWRITER_AVAILABLE

        if not XLSXWRITER_AVAILABLE:
            pytest.skip("xlsxwriter not available")

        import polars as pl

        df = pl.DataFrame({"A": [1, 2, 3], "B": ["x", "y", "z"]})
        output_path = tmp_path / "output.xlsx"

        polars_engine.write_excel(df, output_path)

        assert output_path.exists()
