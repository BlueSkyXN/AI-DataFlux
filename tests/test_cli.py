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
        test_version_metadata_consistency 验证跨 Python/Web/文档的版本一致性
        test_release_workflow_version_tags 验证 Release workflow 支持历史和 v 前缀 tag
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

import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from src import __version__

PROJECT_ROOT = Path(__file__).resolve().parents[1]


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
        assert result.stdout.strip() == f"AI-DataFlux v{__version__}"

    def test_version_metadata_consistency(self):
        """测试 Python、Web 包元数据和用户文档使用同一版本"""
        assert re.fullmatch(
            r"\d+\.\d+\.\d+(?:-(?:dev|alpha|beta|rc)(?:\.\d+)?)?",
            __version__,
        )

        package = json.loads(
            (PROJECT_ROOT / "web/package.json").read_text(encoding="utf-8")
        )
        package_lock = json.loads(
            (PROJECT_ROOT / "web/package-lock.json").read_text(encoding="utf-8")
        )
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        build_docs = (PROJECT_ROOT / "docs/BUILD_VARIANTS.md").read_text(
            encoding="utf-8"
        )

        assert package["version"] == __version__
        assert package_lock["version"] == __version__
        assert package_lock["packages"][""]["version"] == __version__
        assert f"## AI-DataFlux {__version__}" in readme
        assert f"AI-DataFlux v{__version__}" in build_docs

    def test_release_workflow_version_tags(self):
        """测试 Release workflow 同时支持数字版和 v 前缀 SemVer tag"""
        expected_tags = {
            "v[0-9]*.[0-9]*.[0-9]*",
            "[0-9]*.[0-9]*.[0-9]*",
        }

        for workflow_name in ("build-pyinstaller.yml", "build-nuitka.yml"):
            workflow = yaml.load(
                (PROJECT_ROOT / ".github/workflows" / workflow_name).read_text(
                    encoding="utf-8"
                ),
                Loader=yaml.BaseLoader,
            )

            assert set(workflow["on"]["push"]["tags"]) == expected_tags
            assert workflow["jobs"]["release"]["if"] == (
                "startsWith(github.ref, 'refs/tags/')"
            )
            release_step = next(
                step
                for step in workflow["jobs"]["release"]["steps"]
                if step.get("uses") == "softprops/action-gh-release@v2"
            )
            assert (
                "contains(github.ref_name, 'dev')" in release_step["with"]["prerelease"]
            )

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

    def test_process_validate_rejects_semantic_errors(self, tmp_path):
        """测试 --validate 能拦截运行前置配置错误"""
        invalid_config = tmp_path / "semantic_invalid.yaml"
        invalid_config.write_text(
            """
datasource:
  type: mongodb
  concurrency:
    batch_size: 0
columns_to_extract: []
columns_to_write: {}
prompt:
  template: "{record_json}"
""".strip(),
            encoding="utf-8",
        )

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

        assert result.returncode == 1
        assert "Config invalid" in result.stdout
        assert "datasource.type" in result.stdout
        assert "batch_size" in result.stdout

    def test_main_validate_rejects_semantic_errors(self, tmp_path):
        """测试 main.py --validate 复用同一套语义校验"""
        invalid_config = tmp_path / "main_semantic_invalid.yaml"
        invalid_config.write_text(
            """
datasource:
  type: mongodb
columns_to_extract: []
columns_to_write: {}
prompt:
  template: "{record_json}"
""".strip(),
            encoding="utf-8",
        )

        result = subprocess.run(
            [
                sys.executable,
                "main.py",
                "--config",
                str(invalid_config),
                "--validate",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
        )

        assert result.returncode == 1
        assert "配置文件无效" in result.stdout
        assert "datasource.type" in result.stdout

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
