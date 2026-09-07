# AI-DataFlux 4.0 工作区验证与剩余交付项

## 状态

2026-09-08，`codex/4.0-integration`，基线 HEAD 为 `3d377e1011a29c8d07d59730973e1e8a79ef2469`。H2～H3 的新增实现及 H4 本地产品联调已推进，改动尚未提交；基线 SHA 不是本次修改的接受 SHA。

**本地源码验证通过，不等于 Promotion / Release ready。** H1 真实 MySQL/PostgreSQL、Feishu 以及真实 provider UAT、跨平台 exact-head CI 和发布仍缺少证据。打包验证单独记录，不用编译成功代替执行成功。

## 本地门禁

| 检查 | 实际结果 |
|---|---|
| 非 integration 全量 pytest，包含 coverage | `617 passed, 1 skipped, 13 deselected`；包含最后的冻结程序探测、factory 延迟导入和初始化线程修复 |
| integration pytest | `11 passed, 2 skipped`；两个 skip 是未启用的真实 MySQL/PostgreSQL 测试；包含合并/独立 Supervisor 两种真实本地进程链路 |
| overall line / branch | `80.54% / 67.29%`，通过 `75% / 65%` 门槛 |
| jobs / core-runner / gateway aggregate line | `93.07% / 89.39% / 89.40%`，均通过 `85%` 门槛；关键单文件门槛未降低 |
| Ruff / Black / mypy | 全量检查通过；mypy 默认未注解函数体的提示保留 |
| 前端 ESLint / Vitest / production build | 通过；Vitest `5 files / 9 tests` |
| Playwright | `5 passed`；原有这组测试仍为 mock API，不替代下一节真实联调 |
| `npm audit --package-lock-only --audit-level=high` | `0 vulnerabilities` |
| `pip-audit -r requirements.txt` | `No known vulnerabilities found`；该命令审计 requirements 解析结果，不代表整个本机 Python 环境或每个打包产物的 SBOM 已审计 |
| actionlint / canonical v4 example validate | 通过 |

随后针对 smoke 失败清理补充进程组回收，定向执行 `tests/integration/test_local_workflow.py` 为 `3 passed`（一个清理回归、两种真实本地链路）；Ruff、Black 和 diff whitespace 检查通过。该新增清理回归不计入上述先前全量 pytest 的 617。

可复核命令：

```bash
python3 -m pytest tests/ -q -m 'not integration' --cov=src --cov-branch --cov-report=json:coverage.json
python3 .github/scripts/check_coverage.py coverage.json
env -u DATAFLUX_DB_INTEGRATION python3 -m pytest tests/ -q -m integration -rs
ruff check src/ tests/ cli.py main.py gateway.py .github/scripts
black --check src/ tests/ cli.py main.py gateway.py .github/scripts
mypy src/ --ignore-missing-imports
cd web
npm run lint
npm test
npm run build
npm run test:e2e
```

## 真实本地链路（仍非真实 provider UAT）

新增 `.github/scripts/smoke_workflow.py`：临时目录内创建 CSV、canonical config 和 Repository；启动本地假 provider、真实 Gateway 进程及真实 Control/Supervisor 进程，全部使用 loopback 临时端口，finally 关闭进程并清理临时文件。HTTP 回读绕过代理环境，确保确实请求本机服务。

已通过：

1. Control Job API 无 token 返回 401。
2. 真实 HTTP submit → Worker → Gateway → 本地假上游 → CSV 写回。
3. Job 进入 completed，`counts.persisted=1`，CSV 磁盘内容一致。
4. 独立 CLI `job status --json` 回读同一 Job，与 API 一致。
5. events 包含 `record_persisted`，公开 JSON 不暴露 checkpoint 输入。
6. Codex BrowserUse 在真实页面加载配置，查看 strict/local，切换 auto，通过 Control API 保存并回读新 revision，再提交 Job；页面实际显示 completed、持久化 1、失败 0，随后脚本回读 CSV 与 CLI 再次通过。没有使用 `page.route()` 模拟此链路。

`gui --no-worker` 可让 Control/GUI 与独立 `worker` 进程共用 Repository，而不争抢第二个 Supervisor；非 Supervisor 的资源接口从持久化状态读取 active/queued Job。

```bash
python3 .github/scripts/smoke_workflow.py
# 浏览器人工/工具联调；按 Enter 后自动检查提交结果并关闭环境
python3 .github/scripts/smoke_workflow.py --serve
# 独立 Worker + 不启动 Supervisor 的 Control
python3 .github/scripts/smoke_workflow.py --separate-worker
# 给定 Full 二进制后，使用同一链路检查真实产物
python3 .github/scripts/smoke_workflow.py --binary /path/to/full-binary
```

## 打包验证（尚未通过）

- 启动链路已修复三处源码问题：factory 只在选中数据源后导入原生 DB driver；冻结程序使用受限内部入口进行子进程库探测，不再把自身误当 Python 解释器；processor 构造移到线程，避免阻塞 Control event loop，取消时等待构造并关闭 adapter。这些修复的定向测试和上表全量回归通过，不等于产物 smoke 已通过。
- macOS arm64、Python 3.11.9、PyInstaller 6.19.0 已实际生成 Full 与 CLI-only onefile；CLI-only 的 version、worker help 正常，gui help 被明确拒绝。
- 首轮全局环境产物在 `pkg_resources.NullProvider` runtime hook 失败。本机 setuptools 82 留有空 namespace，临时构建 venv 用 setuptools 80.9.0 覆盖后，version 和配置验证可执行；没有修改全局 Python 依赖。
- Full 的 Gateway/Control 启动链路仍出现 50 秒健康检查超时；绕过本机 HTTP 代理、排除非项目 PyObjC 模块后仍复现。采样观察到 Python 动态扩展加载停在 dyld `mapSegments` / `fcntl`，尚不能据此认定唯一根因。产物的 `codesign --verify --strict` 通过，不代表启动 smoke 通过；没有修改系统安全设置或绕过签名检查。
- 包含最后源码修复的 PyInstaller onedir 诊断产物，首次 Gateway 启动卡在 setuptools runtime hook 导入原生 `_heapq`；随后单独加载原始与打包 `_heapq` 均成功（低于 0.01 秒），重跑目录版合并/独立 Supervisor 两种完整链路均得到 `PASS / completed / persisted=1`。这是真实执行通过，不证明首次冷启动稳定，也不替代 onefile 门禁。
- 已从当前源码重建 Full onefile（不带诊断 runtime hook），Gateway/Control 可启动，但提交 Job 后 HTTP 状态读取仍出现 5 秒超时，完整 smoke 失败。没有提高超时阈值、关闭系统保护或将目录版改成正式发布格式来消除失败。
- 全新隔离环境安装分别遇到 MySQL wheel 与 Polars runtime wheel 下载重试超时。没有因此移除项目的数据源或引擎支持。
- Nuitka 4.2.1 首轮发现全局环境中并非本项目依赖的 Foundation，并拒绝 console onefile；排除 Foundation/AppKit/Cocoa/objc 和无关开发/绘图库后已完成编译。随后根据最后源码修复再次增量构建（2504 个 C cache hit、6 个 miss），当前 standalone 目录产物真实链路为 `PASS / completed / persisted=1`。此本地诊断参数未被冒充成 exact-head CI 结果。
- 当前 Nuitka onefile 也实际执行了完整链路：Gateway/Control 启动成功，提交后 Job 状态 HTTP 读取超时，smoke 失败。两个构建器的目录版与 onefile 表现存在差异，但现有证据不足以认定唯一根因；不能只根据目录版通过就把 onefile 标为通过。测试失败后的进程组已回收，没有保留运行服务。
- 在当前 PyInstaller onefile 的失败现场进行原生线程采样，进一步确认了阻塞机制：processor 初始化线程位于 `_imp_create_dynamic → dlopen → dyld::mapSegments → fcntl`，同时 Control 的 uvloop 主线程停在 `PyGILState_Ensure → take_gil`。因此 `asyncio.to_thread` 只能移开普通同步构造，不能隔离持有 GIL 的原生扩展加载；这次 HTTP 超时不是只凭猜测归因于网络或 Repository 锁。尚未证明 dyld 底层等待的唯一原因，下一步需要干净构建环境或对应提交的 CI 复核；不通过关闭系统保护、提高超时或把目录版替换为发布格式消除门禁失败。
- 本地构建都不是 Release asset，也不能证明 Linux/Windows matrix。需要以实际二进制执行结果更新本节。

## 仍需完成的正式交付

1. 将当前源码提交后，由对应 commit 的 CI 生成全部平台/变体并执行 smoke；用干净 runner 复核本机 onefile 超时，按失败日志继续修复，不能以本机结果替代跨平台验收。
2. 使用 Test 工作流现有的 disposable MySQL 8.0 / PostgreSQL 17 service containers 完成真实数据库合同测试，无需用户另外提供数据库实例；Feishu 和真实 provider UAT 仍需专用测试配置及写入授权。
3. 形成可接受的本地 commit，将完整 SHA 与证据回填 Ledger；本次未执行 commit、push、PR、tag 或 Release。
4. 用户明确授权后才进行远端推送、PR、发布和部署；部署后还需版本回读与业务 UAT。

新旧 Repository schema 及锁实现不能混跑；本次没有迁移或覆盖用户现有 Job 数据。原始 H1/H2 切片快照只作历史依据，当前行为以本页、JOBS 和 GATEWAY_API 为准。

## CI 接入状态

2026-09-08 已通过 GitHub CLI 确认 `Test`、`Build with PyInstaller`、`Build with Nuitka` 均为 active；远端尚无 `codex/4.0-integration` 分支，该分支的 workflow run 列表为空，因此不能声称 CI 已验证本轮改动。

- 保留现有测试与打包平台矩阵，以及仅 tag 才执行 Release 的条件。
- 在 PyInstaller Full 与 Nuitka 的各平台构建步骤后，新增产物真实处理链路：合并 Supervisor、独立 Worker 两种模式，均在上传产物前执行。
- 新 smoke 不只检查 `/health`，还验证 Job 提交、Gateway 请求、CSV 写回和 CLI JSON 回读；失败时输出临时测试服务日志，回收测试进程。
- 已通过本地 `actionlint`、Ruff、Black、`git diff --check` 和 `tests/integration/test_local_workflow.py`（`3 passed`）。这些是 CI 配置与脚本的本地验证，不是远端运行结果。
- 推送授权后可在该研发分支手动触发三条工作流，不需要为了测试创建 tag、合并主分支或发布 Release。
