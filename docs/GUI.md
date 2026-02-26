# Web GUI 控制面板

AI-DataFlux 提供了一个本地 Web GUI 控制面板，用于管理 Gateway 和 Process 服务。

## 功能概述

控制面板提供以下核心功能：

1. **Dashboard - 进程管理**
   - 启动/停止 Gateway 和 Process 服务
   - 实时状态监控（PID、端口、模型数、运行时间、进度等）
   - 进度条显示任务处理进度
   - 支持外部进程检测（External标签）
   - 错误计数和退出码显示

2. **配置编辑**
   - 在线编辑 `config.yaml` 配置文件
   - YAML 语法验证
   - 自动备份（.bak文件）
   - 未保存更改提示

3. **日志查看**
   - 实时查看 Gateway 和 Process 的日志输出
   - 左右分栏同时显示两个服务的日志
   - WebSocket 自动重连（指数退避策略）
   - 支持日志复制和清空
   - 自动滚动控制

4. **多语言支持**
   - 界面支持中文/英文切换
   - 自动保存语言偏好设置

5. **工作目录显示**
   - 实时显示当前工作目录
   - 帮助定位配置文件和数据文件

## 快速开始

### 启动控制面板

```bash
# 启动控制面板（自动打开浏览器）
python cli.py gui

# 指定端口
python cli.py gui --port 8080

# 不自动打开浏览器
python cli.py gui --no-browser
```

控制面板默认运行在 `http://127.0.0.1:8790`。

### 命令行参数

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `-p, --port` | 控制服务器端口 | 8790 |
| `--no-browser` | 不自动打开浏览器 | False |

## 架构设计

```
python cli.py gui
       │
       ▼
┌─────────────────────────────────┐
│  Control Server (FastAPI)       │
│  127.0.0.1:8790                 │
│                                 │
│  ┌───────────┐ ┌──────────────┐ │
│  │ ConfigAPI │ │ProcessManager│ │
│  └───────────┘ └──────┬───────┘ │
│                       │         │
│  ┌────────────────────┤         │
│  │ LogBroker (WS)     │         │
│  └────────────────────┤         │
│         serve web/dist/         │
└─────────────────────────────────┘
          │ subprocess        │ subprocess
          ▼                   ▼
   ┌──────────────┐   ┌───────────────┐
   │ Gateway      │   │ Process       │
   │ (port 8787)  │   │ (cli.py       │
   │              │   │  process ...) │
   └──────────────┘   └───────────────┘
```

### 关键设计决策

- **启动方式**：`python cli.py gui` 一条命令启动 Control Server，自动打开浏览器
- **端口分配**：Control Server 使用 `8790`，Gateway 保持 `8787`
- **前端托管**：FastAPI 直接 serve `web/dist/` 静态文件
- **退出行为**：Ctrl+C 关闭 Control Server 时，子进程跟着停止
- **仅本地访问**：只监听 127.0.0.1，且 API/WS 默认启用 Bearer Token 鉴权

## API 接口

控制面板提供以下 REST API 接口：

> 认证说明：所有 `/api/*` 请求都需要 `Authorization: Bearer <token>`。
> `python cli.py gui` 自动打开浏览器时会携带 `#token=...`，前端会自动透传。

### 配置文件 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/config?path=config.yaml` | 读取配置文件内容（仅允许 `.yaml/.yml`） |
| `PUT` | `/api/config` | 写入配置文件（自动备份） |
| `POST` | `/api/config/validate` | 校验 YAML 语法（仅解析，不做业务校验） |

**PUT /api/config 请求体**:
```json
{
  "path": "config.yaml",
  "content": "# YAML content..."
}
```

**POST /api/config/validate 请求体**:
```json
{
  "content": "# YAML content..."
}
```

**POST /api/config/validate 响应**:

成功：
```json
{
  "valid": true
}
```

失败：
```json
{
  "valid": false,
  "error": "..."
}
```

### 进程管理 API

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/gateway/start` | 启动 Gateway |
| `POST` | `/api/gateway/stop` | 停止 Gateway |
| `POST` | `/api/process/start` | 启动 Process |
| `POST` | `/api/process/stop` | 停止 Process |
| `GET` | `/api/status` | 获取所有进程状态 |

> 说明：为缓解 localhost CSRF，所有写操作（`POST/PUT/PATCH/DELETE` 且路径为 `/api/*`）都要求 `Content-Type: application/json`（否则返回 415）。

**POST /api/gateway/start 请求体** (可选):
```json
{
  "config_path": "config.yaml",
  "port": 8787,
  "workers": 1
}
```

**POST /api/process/start 请求体** (可选):
```json
{
  "config_path": "config.yaml"
}
```

### 日志流 API

| 协议 | 路径 | 说明 |
|------|------|------|
| `WS` | `/api/logs?target=gateway` | Gateway 日志流（鉴权 token 通过 `Sec-WebSocket-Protocol: dataflux-token-b64.<base64url>` 透传） |
| `WS` | `/api/logs?target=process` | Process 日志流（鉴权 token 通过 `Sec-WebSocket-Protocol: dataflux-token-b64.<base64url>` 透传） |

### 响应示例

`GET /api/status` 响应：

```json
{
  "gateway": {
    "managed": {
      "status": "running",
      "pid": 12345,
      "start_time": 1707000000,
      "config_path": "config.yaml",
      "port": 8787
    },
    "health": {
      "status": "healthy",
      "available_models": 3,
      "total_models": 3,
      "uptime": 3600.5
    }
  },
  "process": {
    "managed": {
      "status": "running",
      "pid": 12346,
      "start_time": 1707000100,
      "config_path": "config.yaml"
    },
    "progress": {
      "total": 2000,
      "processed": 847,
      "active": 5,
      "shard": "2/4",
      "errors": 3
    }
  },
  "working_directory": "/path/to/project"
}
```

## 进程状态

进程有三种状态：

| 状态 | 说明 |
|------|------|
| `stopped` | 进程未运行 |
| `running` | 进程正在运行 |
| `exited` | 进程已退出（可能成功或失败） |

状态转换图：

```
STOPPED  ──start──▶  RUNNING  ──进程退出──▶  EXITED
   ▲                    │                      │
   └────────stop────────┘                      │
   └───────────────────────────────────────────┘
```

## 配置文件编辑

配置编辑器支持：

- **原始文本编辑**：保留 YAML 注释和格式
- **自动备份**：写入前自动创建 `.bak` 备份文件
- **原子写入**：通过临时文件确保写入完整性
- **路径保护**：防止路径穿越攻击
- **写入白名单**：仅允许写入 `.yaml/.yml` 配置文件

## 日志查看

日志窗口功能：

- **实时流式日志**：通过 WebSocket 实时推送
- **历史日志缓存**：保留最近 1000 行日志
- **stdout/stderr 合并**：统一显示输出和错误

## 进度监控

当 Process 运行时，Dashboard 会显示：

- **总任务数**：待处理的任务总数
- **已处理数**：成功处理的任务数
- **活跃任务**：正在处理中的任务数
- **当前分片**：当前处理的分片位置
- **错误数**：重试失败的任务数

进度数据通过 `--progress-file` 参数传递，每 5 秒更新一次。

## 开发说明

### 前端开发

前端使用 React + TypeScript + Vite + Tailwind CSS 构建。

```bash
# 进入前端目录
cd web

# 安装依赖
npm install

# 开发模式（自动代理 API 到后端）
npm run dev

# 构建生产版本
npm run build
```

开发模式下，Vite 会自动将 `/api` 请求代理到 `http://127.0.0.1:8790`。
（包含 WebSocket：`/api/logs`）
如需指定其他后端端口，可在启动前端时设置环境变量：

```bash
VITE_CONTROL_SERVER=http://127.0.0.1:9000 npm run dev
```

Windows 下可使用 PowerShell：
```powershell
$env:VITE_CONTROL_SERVER="http://127.0.0.1:9000"; npm run dev
```

### 后端开发

控制服务器代码位于 `src/control/` 目录：

| 文件 | 说明 |
|------|------|
| `__init__.py` | 模块入口 |
| `server.py` | FastAPI 应用主文件 |
| `config_api.py` | 配置文件读写 API |
| `process_manager.py` | 进程生命周期管理 |
| `runtime.py` | 运行时/打包环境路径解析工具 |

### 环境变量（可选）

| 环境变量 | 说明 |
|---------|------|
| `VITE_CONTROL_SERVER` | 前端开发时，Vite 代理的 Control Server 地址 |
| `DATAFLUX_GUI_CORS_ORIGINS` | 逗号分隔的允许跨域来源（开发调试用） |
| `DATAFLUX_CONTROL_TOKEN` | 控制面鉴权 Token；未设置时服务会自动生成临时 token |
| `DATAFLUX_PROJECT_ROOT` / `AI_DATAFLUX_PROJECT_ROOT` | 覆盖“项目根目录”（打包运行或非仓库目录启动时有用） |

示例：

```bash
# 覆盖项目根目录（配置读写 / 子进程 cwd 都基于此目录）
DATAFLUX_PROJECT_ROOT=/path/to/project python cli.py gui

# 仅当你不使用 Vite proxy、需要跨域直连后端时才需要 CORS
DATAFLUX_GUI_CORS_ORIGINS=http://127.0.0.1:5173 python cli.py gui --no-browser
```

## 跨平台支持

控制面板支持以下平台：

- Linux (x64/ARM64)
- macOS (x64/ARM64)
- Windows (x64)

进程管理使用 `psutil` 库实现跨平台的进程树管理，确保子进程能被正确清理。

## 安全说明

- 控制面板仅监听 `127.0.0.1`，不对外暴露
- 控制面 API 与日志 WebSocket 默认要求 Bearer Token 鉴权
- 配置文件路径经过校验，防止目录穿越
- 配置写入仅允许 `.yaml/.yml` 文件，阻断脚本类文件改写
- 配置读取仅允许 `.yaml/.yml` 文件，减少非必要文件暴露面
- 写操作要求 `Content-Type: application/json`，用于降低 localhost CSRF 风险
- 建议不要在公网环境运行

## 常见问题

### Q: 为什么浏览器打开后显示"前端未构建"？

A: 需要先构建前端：

```bash
cd web
npm install
npm run build
```

### Q: 如何修改 Gateway 端口？

A: 在 Dashboard 页面启动 Gateway 前，可以在配置中修改端口，或者通过 API 指定：

```bash
curl -X POST http://127.0.0.1:8790/api/gateway/start \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"port": 8888}'
```

### Q: 调用 API 返回 415 "Content-Type must be application/json"？

A: 出于 localhost CSRF 缓解策略，所有写操作都必须带 `Content-Type: application/json`（即便没有请求体）。例如：

```bash
curl -X POST http://127.0.0.1:8790/api/gateway/stop \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{}'
```

### Q: 日志显示"连接断开"怎么办？

A: 页面会自动重连；也可以点击“Reconnect”按钮，或刷新页面重新建立 WebSocket 连接。
