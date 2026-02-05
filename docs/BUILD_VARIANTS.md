# 构建版本说明

AI-DataFlux 提供两种构建版本以满足不同使用场景：

## 版本类型

### 1. 完整版 (Full)

**包含组件**：
- ✅ 核心处理引擎（process）
- ✅ API 网关（gateway）
- ✅ Token 估算工具（token）
- ✅ Web GUI 控制面板（gui）
- ✅ 前端静态资源（web/dist）

**适用场景**：
- 需要图形化界面管理 Gateway 和 Process
- 需要实时查看日志和配置编辑器
- 本地开发和测试

**文件命名**：
- Linux x64: `AI-DataFlux-linux-amd64-full`
- Linux ARM64: `AI-DataFlux-linux-arm64-full`
- macOS ARM64: `AI-DataFlux-macos-arm64-full`
- Windows x64: `AI-DataFlux-windows-amd64-full.exe`

### 2. CLI 版本 (CLI-only)

**包含组件**：
- ✅ 核心处理引擎（process）
- ✅ API 网关（gateway）
- ✅ Token 估算工具（token）
- ❌ Web GUI 控制面板（不包含）
- ❌ 前端静态资源（不包含）

**优势**：
- 体积更小（约减少 50% 大小）
- 无需 Node.js 构建依赖
- 适合服务器部署和 CI/CD 环境

**适用场景**：
- 生产环境批处理任务
- 容器化部署
- 自动化脚本调用
- 无需图形界面的场景

**文件命名**：
- Linux x64: `AI-DataFlux-linux-amd64-cli`
- Linux ARM64: `AI-DataFlux-linux-arm64-cli`
- macOS ARM64: `AI-DataFlux-macos-arm64-cli`
- Windows x64: `AI-DataFlux-windows-amd64-cli.exe`

## 功能对比

| 功能 | Full 版 | CLI 版 |
|------|---------|--------|
| `process` 子命令 | ✅ | ✅ |
| `gateway` 子命令 | ✅ | ✅ |
| `token` 子命令 | ✅ | ✅ |
| `version` 子命令 | ✅ | ✅ |
| `check` 子命令 | ✅ | ✅ |
| `gui` 子命令 | ✅ | ❌ |
| Web 控制面板 | ✅ | ❌ |
| 配置编辑器 | ✅ | ❌ |
| 实时日志查看 | ✅ | ❌ |

## 技术实现

### CLI 自适应

CLI 在启动时会自动检测 `src.control.server` 模块是否可用：

```python
# main() 函数中的条件性注册
try:
    gui_available = importlib.util.find_spec("src.control.server") is not None
except (ModuleNotFoundError, ImportError):
    gui_available = False

if gui_available:
    # 注册 gui 子命令
    p_gui = subparsers.add_parser("gui", help="Start GUI control panel")
```

**行为**：
- **Full 版**：`--help` 显示 `gui` 子命令，可正常使用
- **CLI 版**：`--help` 不显示 `gui` 子命令，避免用户困惑

### 打包差异

#### Full 版构建流程
```bash
# 1. 构建前端
cd web && npm ci && npm run build

# 2. 打包（包含 web/dist）
pyinstaller --onefile --clean \
  --name AI-DataFlux \
  --add-data "web/dist:web/dist" \
  cli.py
```

#### CLI-only 版构建流程
```bash
# 1. 跳过前端构建

# 2. 打包（排除 src.control）
pyinstaller --onefile --clean \
  --name AI-DataFlux \
  --exclude-module=src.control \
  cli.py
```

**关键参数**：
- `--add-data`: 打包静态资源（仅 Full 版）
- `--exclude-module`: 排除 GUI 控制器模块（仅 CLI 版）

## 版本选择建议

### 选择 Full 版
- ✅ 首次使用，需要学习和测试
- ✅ 本地开发调试
- ✅ 需要可视化界面
- ✅ 配置文件频繁修改

### 选择 CLI 版
- ✅ 生产环境部署
- ✅ 自动化脚本集成
- ✅ 容器化环境（如 Docker）
- ✅ 对包体积敏感
- ✅ 纯命令行操作习惯

## 验证版本

### 验证 Full 版
```bash
# 检查 gui 子命令是否存在
./AI-DataFlux-linux-amd64-full --help | grep gui
# 输出: gui  Start GUI control panel

# 尝试启动 GUI
./AI-DataFlux-linux-amd64-full gui --no-browser
# 输出: Control server started at http://127.0.0.1:8790
```

### 验证 CLI 版
```bash
# 检查 gui 子命令不存在
./AI-DataFlux-linux-amd64-cli --help | grep gui
# 无输出（grep 返回非零退出码）

# 验证核心功能可用
./AI-DataFlux-linux-amd64-cli version
# 输出: AI-DataFlux v2.0.0

./AI-DataFlux-linux-amd64-cli check
# 输出: 库状态信息
```

## CI/CD 集成

GitHub Actions workflow 会自动构建两个版本：

```yaml
matrix:
  build_type: [full, cli-only]
  os: [ubuntu-24.04, ubuntu-24.04-arm, macos-15, windows-2025]
```

每次 tag 推送或 main 分支提交时，会产出 8 个 artifact（4 平台 × 2 版本）。

## 常见问题

### Q: CLI 版能否手动添加 GUI？
A: 不能。CLI 版在构建时已排除 `src.control` 模块和前端资源，无法通过后期添加文件恢复功能。需要下载 Full 版。

### Q: Full 版能否禁用 GUI？
A: 可以。只需不使用 `gui` 子命令即可，其他功能完全独立。

### Q: 两个版本性能有差异吗？
A: 核心处理性能完全一致。CLI 版仅在启动时略快（因为不加载 GUI 相关模块）。

### Q: 如何从 CLI 版升级到 Full 版？
A: 直接下载 Full 版替换可执行文件即可，配置文件和数据完全兼容。

## 开发者说明

### 本地测试两种版本

#### 测试 Full 版行为
```bash
python cli.py --help  # 应显示 gui 子命令
python cli.py gui --no-browser  # 应正常启动
```

#### 模拟 CLI 版行为
```bash
# 临时重命名 src.control 目录
mv src/control src/control.bak

# 测试
python cli.py --help  # 不应显示 gui 子命令
python cli.py version  # 其他功能正常

# 恢复
mv src/control.bak src/control
```

### 添加新的可选功能

如需添加其他可选特性（类似 GUI 的拆分方式）：

1. **隔离模块**：将功能代码放在独立目录（如 `src/optional_feature/`）
2. **条件导入**：CLI 注册时检查模块可用性
3. **防御式导入**：命令函数内用 try-except 处理 ImportError
4. **打包排除**：在 CLI-only 构建时添加 `--exclude-module=src.optional_feature`

参考 `cli.py:513-526` 和 `cli.py:411-421` 的实现。
