"""
CLI 入口测试

测试 cli.py 的各个子命令
"""

import subprocess
import sys


class TestCLI:
    """CLI 命令测试"""

    def test_version(self):
        """测试 version 命令"""
        result = subprocess.run(
            [sys.executable, "cli.py", "version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "AI-DataFlux" in result.stdout
        assert "v" in result.stdout.lower() or "2." in result.stdout

    def test_check(self):
        """测试 check 命令"""
        result = subprocess.run(
            [sys.executable, "cli.py", "check"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "pandas" in result.stdout
        assert "openpyxl" in result.stdout

    def test_help(self):
        """测试帮助信息"""
        result = subprocess.run(
            [sys.executable, "cli.py", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "process" in result.stdout
        assert "gateway" in result.stdout
        assert "token" in result.stdout
        assert "version" in result.stdout
        assert "check" in result.stdout

    def test_process_help(self):
        """测试 process 子命令帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py", "process", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--validate" in result.stdout

    def test_gateway_help(self):
        """测试 gateway 子命令帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py", "gateway", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--host" in result.stdout

    def test_process_validate(self, sample_config_file):
        """测试配置验证"""
        result = subprocess.run(
            [
                sys.executable,
                "cli.py",
                "process",
                "--config",
                str(sample_config_file),
                "--validate",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "Config valid" in result.stdout or "[OK]" in result.stdout

    def test_process_invalid_config(self, tmp_path):
        """测试无效配置文件"""
        invalid_config = tmp_path / "invalid.yaml"
        invalid_config.write_text("invalid: yaml: content:")

        result = subprocess.run(
            [
                sys.executable,
                "cli.py",
                "process",
                "--config",
                str(invalid_config),
                "--validate",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0

    def test_no_command(self):
        """测试无命令时显示帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # 应该显示帮助信息
        assert "usage" in result.stdout.lower() or "help" in result.stdout.lower()

    def test_token_help(self):
        """测试 token 子命令帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py", "token", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--mode" in result.stdout
        # 验证新的模式选项
        assert "in" in result.stdout
        assert "out" in result.stdout
        assert "io" in result.stdout
