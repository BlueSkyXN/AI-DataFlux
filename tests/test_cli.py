"""
CLI 入口测试

被测模块: cli.py

测试 cli.py 的各个子命令，包括：
- version: 版本信息显示
- check: 依赖库检查
- help: 帮助信息
- process: 主处理流程 (配置验证)
- token: Token 估算命令
- gui: Web GUI 控制面板

测试类/函数清单:
    TestCLI                        CLI 命令测试
        test_version               验证 version 命令输出包含版本号
        test_check                 验证 check 命令输出包含依赖库名
        test_help                  验证 --help 列出所有子命令
        test_process_help          验证 process --help 显示 --config/--validate 参数
        test_gateway_help          验证 gateway --help 显示 --port/--host 参数
        test_gui_help              验证 gui --help 显示 --port/--no-browser 参数
        test_process_validate      验证有效配置文件通过 --validate 检查
        test_process_invalid_config 验证无效配置文件导致非零退出码
        test_no_command            验证无命令时显示帮助信息
        test_token_help            验证 token --help 显示 --mode 及模式选项
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
            encoding="utf-8",
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
            encoding="utf-8",
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
            encoding="utf-8",
        )
        assert result.returncode == 0
        assert "process" in result.stdout
        assert "gateway" in result.stdout
        assert "token" in result.stdout
        assert "version" in result.stdout
        assert "check" in result.stdout
        assert "gui" in result.stdout

    def test_process_help(self):
        """测试 process 子命令帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py", "process", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
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
            encoding="utf-8",
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--host" in result.stdout

    def test_gui_help(self):
        """测试 gui 子命令帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py", "gui", "--help"],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )
        assert result.returncode == 0
        assert "--port" in result.stdout
        assert "--no-browser" in result.stdout

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
            encoding="utf-8",
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
            encoding="utf-8",
        )
        assert result.returncode != 0

    def test_no_command(self):
        """测试无命令时显示帮助"""
        result = subprocess.run(
            [sys.executable, "cli.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",
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
            encoding="utf-8",
        )
        assert result.returncode == 0
        assert "--config" in result.stdout
        assert "--mode" in result.stdout
        # 验证新的模式选项
        assert "in" in result.stdout
        assert "out" in result.stdout
        assert "io" in result.stdout
