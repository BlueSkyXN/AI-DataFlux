"""
Control Server 测试

测试 Web GUI 控制面板的核心功能：
- 配置文件 API (读取/写入)
- 进程管理 (状态查询)
- 路径安全验证
"""

from unittest.mock import patch

import pytest


class TestConfigAPI:
    """配置文件 API 测试"""

    def test_validate_path_inside_project(self):
        """测试路径校验 - 项目内部路径"""
        from src.control.config_api import _validate_path, PROJECT_ROOT

        # 项目内部路径应该通过
        result = _validate_path("config-example.yaml")
        assert result.startswith(PROJECT_ROOT)

    def test_validate_path_traversal_blocked(self):
        """测试路径校验 - 阻止路径穿越"""
        from fastapi import HTTPException
        from src.control.config_api import _validate_path

        # 尝试访问项目外部应该被阻止
        with pytest.raises(HTTPException) as exc_info:
            _validate_path("../../../etc/passwd")

        assert exc_info.value.status_code == 403

    def test_read_config_existing_file(self):
        """测试读取存在的配置文件"""
        from src.control.config_api import read_config

        content = read_config("config-example.yaml")
        assert "global" in content
        assert "datasource" in content

    def test_read_config_missing_file(self):
        """测试读取不存在的配置文件"""
        from fastapi import HTTPException
        from src.control.config_api import read_config

        with pytest.raises(HTTPException) as exc_info:
            read_config("nonexistent.yaml")

        assert exc_info.value.status_code == 404

    def test_write_config_creates_backup(self, tmp_path):
        """测试写入配置文件会创建备份"""
        from src.control.config_api import write_config

        # 使用临时目录中的文件
        test_file = tmp_path / "test_config.yaml"
        test_file.write_text("original: content")

        # Mock PROJECT_ROOT to use tmp_path
        with patch("src.control.config_api.PROJECT_ROOT", str(tmp_path)):
            result = write_config("test_config.yaml", "new: content")

        assert result["success"] is True
        assert result["backed_up"] is True
        assert (tmp_path / "test_config.yaml.bak").exists()
        assert (tmp_path / "test_config.yaml").read_text() == "new: content"


class TestProcessManager:
    """进程管理器测试"""

    def test_initial_status(self):
        """测试初始状态"""
        from src.control.process_manager import ProcessManager

        manager = ProcessManager()

        gateway_status = manager.get_status("gateway")
        process_status = manager.get_status("process")

        assert gateway_status["status"] == "stopped"
        assert gateway_status["pid"] is None
        assert process_status["status"] == "stopped"
        assert process_status["pid"] is None

    def test_get_all_status(self):
        """测试获取所有状态"""
        from src.control.process_manager import ProcessManager

        manager = ProcessManager()
        all_status = manager.get_all_status()

        assert "gateway" in all_status
        assert "process" in all_status
        assert "managed" in all_status["gateway"]
        assert "managed" in all_status["process"]

    def test_log_buffer(self):
        """测试日志缓冲区"""
        from src.control.process_manager import ProcessManager

        manager = ProcessManager()

        # 初始应该为空
        gateway_logs = manager.get_log_buffer("gateway")
        assert gateway_logs == []

        process_logs = manager.get_log_buffer("process")
        assert process_logs == []


class TestManagedProcess:
    """ManagedProcess 数据类测试"""

    def test_to_dict(self):
        """测试转换为字典"""
        from src.control.process_manager import ManagedProcess

        proc = ManagedProcess(
            name="gateway",
            status="running",
            pid=12345,
            start_time=1707000000.0,
            config_path="config.yaml",
            port=8787,
        )

        result = proc.to_dict()

        assert result["status"] == "running"
        assert result["pid"] == 12345
        assert result["start_time"] == 1707000000.0
        assert result["config_path"] == "config.yaml"
        assert result["port"] == 8787

    def test_default_values(self):
        """测试默认值"""
        from src.control.process_manager import ManagedProcess

        proc = ManagedProcess(name="process")

        assert proc.status == "stopped"
        assert proc.pid is None
        assert proc.start_time is None
        assert proc.exit_code is None
        assert proc.command == []
        assert proc.config_path is None
        assert proc.port is None


class TestKillTree:
    """进程树清理测试"""

    def test_kill_tree_nonexistent_process(self):
        """测试清理不存在的进程"""
        from src.control.process_manager import kill_tree

        # 不应该抛出异常
        kill_tree(99999999)
