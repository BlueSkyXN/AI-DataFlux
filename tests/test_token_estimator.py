"""
Token 估算器测试

被测模块: src/core/token_estimator.py (TokenEstimator, normalize_mode)

测试 src/core/token_estimator.py 的核心功能，包括：
- 模式规范化 (in/out/io)
- Token 计数算法
- tiktoken 编码选择
- 批量估算性能

测试类/函数清单:
    TestNormalizeMode                              模式规范化测试
        test_normalize_legacy_input               验证 "input" → "in"
        test_normalize_legacy_input_output        验证 "input_output" → "io"
        test_normalize_new_modes                  验证 in/out/io 保持不变
        test_normalize_case_insensitive           验证大小写不敏感
        test_normalize_unknown_mode               验证未知模式回退到 "in"
    TestTokenEstimator                            Token 估算器测试
        test_create_prompt                        验证提示词模板替换
        test_build_messages                       验证消息构建（system + user）
        test_count_message_tokens_uses_combined_text  验证消息合并后计数
        test_estimate_output_tokens_for_record    验证输出 token 估算
        test_compute_stats                        验证统计计算（avg/min/max/total）
        test_tiktoken_not_available               验证 tiktoken 不可用时抛 ImportError
        test_mode_out_skips_input_estimation      验证 out 模式跳过输入估算
        test_mode_io_estimates_both               验证 io 模式同时估算输入和输出
    TestTokenEstimatorIntegration                  集成测试
        test_sample_unprocessed_rows              验证从 Excel 采样未处理行
"""

import pytest
from unittest.mock import MagicMock, patch


class TestNormalizeMode:
    """测试模式规范化函数"""

    def test_normalize_legacy_input(self):
        """测试旧模式 input 转换为 in"""
        from src.core.token_estimator import normalize_mode

        assert normalize_mode("input") == "in"

    def test_normalize_legacy_input_output(self):
        """测试旧模式 input_output 转换为 io"""
        from src.core.token_estimator import normalize_mode

        assert normalize_mode("input_output") == "io"

    def test_normalize_new_modes(self):
        """测试新模式保持不变"""
        from src.core.token_estimator import normalize_mode

        assert normalize_mode("in") == "in"
        assert normalize_mode("out") == "out"
        assert normalize_mode("io") == "io"

    def test_normalize_case_insensitive(self):
        """测试大小写不敏感"""
        from src.core.token_estimator import normalize_mode

        assert normalize_mode("IN") == "in"
        assert normalize_mode("OUT") == "out"
        assert normalize_mode("IO") == "io"

    def test_normalize_unknown_mode(self):
        """测试未知模式回退到 in"""
        from src.core.token_estimator import normalize_mode

        assert normalize_mode("unknown") == "in"


class TestTokenEstimator:
    """Token 估算器测试"""

    @pytest.fixture
    def token_config(self, sample_config):
        """扩展示例配置，添加 token 估算配置"""
        config = sample_config.copy()
        config["token_estimation"] = {
            "mode": "in",
            "sample_size": 10,
            "tiktoken_model": "gpt-4",
        }
        config["prompt"]["system_prompt"] = "You are a helpful assistant."
        config["prompt"]["template"] = "Analyze this: {record_json}"
        return config

    @pytest.fixture
    def mock_tiktoken(self):
        """模拟 tiktoken"""
        with patch.dict("sys.modules", {"tiktoken": MagicMock()}):
            import sys

            mock_tiktoken = sys.modules["tiktoken"]

            # 创建模拟编码器
            mock_encoding = MagicMock()
            mock_encoding.encode.side_effect = lambda x: list(range(len(x) // 4 + 1))

            mock_tiktoken.encoding_for_model.return_value = mock_encoding
            mock_tiktoken.get_encoding.return_value = mock_encoding

            yield mock_tiktoken

    def test_create_prompt(self, token_config, mock_tiktoken):
        """测试提示词创建"""
        # 需要重新导入以使用 mock
        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)

                record_data = {
                    "question": "What is AI?",
                    "context": "AI is artificial intelligence",
                }
                prompt = estimator.create_prompt(record_data)

                assert "Analyze this:" in prompt
                assert "question" in prompt
                assert "What is AI?" in prompt

    def test_build_messages(self, token_config, mock_tiktoken):
        """测试消息构建"""
        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)

                messages = estimator.build_messages("Test prompt")

                assert len(messages) == 2
                assert messages[0]["role"] == "system"
                assert messages[1]["role"] == "user"
                assert messages[1]["content"] == "Test prompt"

    def test_count_message_tokens_uses_combined_text(self, token_config, mock_tiktoken):
        """测试消息合并后再计数"""
        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)

                messages = [
                    {"role": "system", "content": "SYS"},
                    {"role": "user", "content": "USR"},
                ]
                tokens = estimator.count_message_tokens(messages)

                assert tokens > 0
                estimator.encoding.encode.assert_called_with("SYS\nUSR")

    def test_estimate_output_tokens_for_record(self, token_config, mock_tiktoken):
        """测试输出 token 估算"""
        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)

                # 输出数据使用实际列名 (columns_to_write 的值)
                output_data = {"ai_answer": "This is the answer"}
                tokens = estimator.estimate_output_tokens_for_record(output_data)

                # Mock 编码器返回的 token 数量
                assert tokens > 0

    def test_compute_stats(self, token_config, mock_tiktoken):
        """测试统计计算"""
        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)

                tokens_list = [10, 20, 30, 40, 50]
                stats = estimator._compute_stats(tokens_list, 100)

                assert stats["avg"] == 30.0
                assert stats["min"] == 10
                assert stats["max"] == 50
                assert stats["total_estimated"] == 3000  # 30 * 100
                assert stats["sample_count"] == 5

    def test_tiktoken_not_available(self, token_config):
        """测试 tiktoken 不可用时的错误处理"""
        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", False):
            from src.core.token_estimator import TokenEstimator

            with pytest.raises(ImportError) as exc_info:
                TokenEstimator(token_config)

            assert "tiktoken" in str(exc_info.value)

    def test_mode_out_skips_input_estimation(self, token_config, mock_tiktoken):
        """测试 out 模式跳过输入估算"""
        token_config["token_estimation"]["mode"] = "out"

        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)
                assert estimator.mode == "out"

                # 创建 mock 任务池
                mock_pool = MagicMock()
                mock_pool.get_total_task_count.return_value = 10
                mock_pool.get_processed_task_count.return_value = 3
                mock_pool.sample_processed_rows.return_value = [
                    {"ai_answer": "Answer 1"},
                    {"ai_answer": "Answer 2"},
                ]

                result = estimator.estimate(mock_pool, mock_pool)

                # out 模式不应有 input_tokens
                assert "input_tokens" not in result
                # 应有 output_tokens
                assert "output_tokens" in result
                assert "error" not in result.get("output_tokens", {})
                assert result.get("processed_total_rows") == 3
                assert result.get("total_rows") == 3
                assert result.get("request_count") == 3
                assert result.get("output_tokens", {}).get("total_estimated") > 0

    def test_mode_io_estimates_both(self, token_config, mock_tiktoken):
        """测试 io 模式同时估算输入和输出"""
        token_config["token_estimation"]["mode"] = "io"

        with patch("src.core.token_estimator.TIKTOKEN_AVAILABLE", True):
            with patch("src.core.token_estimator.tiktoken", mock_tiktoken):
                from src.core.token_estimator import TokenEstimator

                estimator = TokenEstimator(token_config)
                assert estimator.mode == "io"

                # 创建 mock 输入池
                mock_input_pool = MagicMock()
                mock_input_pool.get_total_task_count.return_value = 10
                mock_input_pool.sample_unprocessed_rows.return_value = [
                    {"question": "Q1", "context": "C1"},
                    {"question": "Q2", "context": "C2"},
                ]

                # 创建 mock 输出池
                mock_output_pool = MagicMock()
                mock_output_pool.get_processed_task_count.return_value = 2
                mock_output_pool.sample_processed_rows.return_value = [
                    {"ai_answer": "Answer 1"},
                    {"ai_answer": "Answer 2"},
                ]

                result = estimator.estimate(mock_input_pool, mock_output_pool)

                # io 模式应同时有 input_tokens 和 output_tokens
                assert "input_tokens" in result
                assert "output_tokens" in result
                assert "error" not in result.get("input_tokens", {})
                assert "error" not in result.get("output_tokens", {})
                assert result.get("processed_total_rows") == 2
                assert result.get("output_tokens", {}).get("total_estimated") > 0


class TestTokenEstimatorIntegration:
    """Token 估算器集成测试"""

    @pytest.fixture
    def sample_excel_with_data(self, tmp_path):
        """创建带数据的 Excel 文件"""
        pd = pytest.importorskip("pandas")

        df = pd.DataFrame(
            {
                "question": ["Q1", "Q2", "Q3", "Q4", "Q5"],
                "context": ["C1", "C2", "C3", "C4", "C5"],
                "ai_answer": [None, "A2", None, None, None],
            }
        )

        excel_path = tmp_path / "test_token.xlsx"
        df.to_excel(excel_path, index=False, engine="openpyxl")
        return excel_path

    def test_sample_unprocessed_rows(self, sample_excel_with_data, sample_config):
        """测试采样未处理行"""
        from src.data import create_task_pool

        config = sample_config.copy()
        config["excel"]["input_path"] = str(sample_excel_with_data)
        config["excel"]["output_path"] = str(
            sample_excel_with_data.parent / "output.xlsx"
        )

        pool = create_task_pool(
            config, config["columns_to_extract"], config["columns_to_write"]
        )

        samples = pool.sample_unprocessed_rows(10)

        # 应该有 4 行未处理 (Q1, Q3, Q4, Q5 没有 ai_answer)
        assert len(samples) <= 4
        assert all("question" in s for s in samples)
