# AI-DataFlux 4.0 H1 本地 Validation Evidence

## 1. 结论

**H1 代码实现完成，本地可执行门禁通过；H1 外部验收仍被真实 DB/Feishu 证据阻断。**

本证据只覆盖 `codex/4.0-integration` 上 H1.0～H1.4 的本地 source、自动化测试、静态质量、依赖审计和 localhost Control→Gateway source smoke。它不证明远端 CI、真实 MySQL/PostgreSQL、真实 Feishu、打包产物、Release、部署或业务 UAT。

## 2. 验证对象与边界

| 项目 | 值 |
|---|---|
| 验证日期 | 2026-08-31，Asia/Shanghai |
| 活动分支 | `codex/4.0-integration` |
| H1.0 | `d4447c5dee86906bbfc4a2720ca5ed85b3fda234` |
| H1.1 | `cf29b4e013244598e5b1dbc7789cfb0433e24a34` |
| H1.2 | `eb9ebb5d32471c9b54503b57dec23b8418ce200c` |
| H1.3 | `f6cba67fd4c8cd20f53a84ca5c49965f59188b22` |
| H1.4 | `b890799a9485318085217c901223da563730f9ef` |
| 冻结 Bootstrap | `406b72894d504a60bdc1ffec003c3ef60d6a6550` |
| 冻结 Bootstrap tree | `cc6c9fffb7f929bd111c0e2c4fe1783b0306df1c` |

禁止项均未执行：未 `push`，未创建或修改 PR，未创建 tag/Release，未部署，未安装 Docker/Podman，未连接远端 MySQL/PostgreSQL，未调用真实 AI provider，未读取或写入真实 Feishu 数据。

## 3. H1 source implementation 追踪

| Slice | 主要实现 | 本地结论 |
|---|---|---|
| H1.0 | canonical v4 `RootConfig`、严格 `schema_version: 4`、component require/compile helper、execution hash 与 YAML ETag 分离、`4.0.0-dev`、旧 migration 删除 | `REVISED_AND_ACCEPTED`，仅限 H1 schema/version/migration 合同 |
| H1.1 | `TaskSuccess`/`TaskFailure`、source/model/content/system 分类、total-attempt retry、reload 和 source failure fail closed | `REVISED_AND_ACCEPTED` |
| H1.2 | immutable `PreparedResult`、`WritebackReceipt` 校验、`PENDING_COMMIT` 恢复点、reconciliation、`UNRESOLVED_WRITE` 与终态优先级 | `REVISED_AND_ACCEPTED`，不包含 H2 Repository 全面验收 |
| H1.3 | SQLite/MySQL/PostgreSQL 事务 receipt、commit unknown、identifier/参数化、readback reconciliation、MySQL `FOUND_ROWS` | SQLite `REVISED_AND_ACCEPTED`；MySQL/PostgreSQL 真实合同仍 `UNREVIEWED` |
| H1.4 | Excel/CSV 原子替换后磁盘回读、目录 fsync 不明态、文件 reconciliation；Bitable chunk 结果；Sheet record/cell 结果与 reconciliation | Excel/CSV `REVISED_AND_ACCEPTED`；Feishu 真实 UAT 仍 `UNREVIEWED` |

H1.2 只实现 writeback correctness 所需的最小 PreparedResult blob/ref/hash 与恢复提交点。Repository 根格式、完整 locking、Supervisor ownership、lease、全面 crash matrix 仍属于 H2，不因本轮测试通过而接受。

H1.0 只冻结 Gateway route/fallback/affinity schema 并接入现有运行入口。strict/fallback/affinity 的完整运行语义仍属于 H3，不因 Gateway protocol 单元测试或 source smoke 而接受。

## 4. Slice 定向验证

### 4.1 H1.4 文件与 Feishu 合同

~~~bash
pytest tests/test_csv_pool.py tests/test_engines.py \
  tests/test_feishu_client_async.py tests/test_feishu_pool.py -q
~~~

结果：`114 passed`。

覆盖的关键行为：

- Pandas/Polars 与 CSV/Excel 等价写回；
- 未提供的输出列保持原值；
- `os.replace` 前失败为明确拒绝，替换后目录 fsync 失败为 `INDETERMINATE`；
- COMMITTED 之前必须重新读取目标文件并逐字段匹配；
- reconciliation 不读取内存 DataFrame 充当磁盘证据；
- Bitable 网络结果不明、明确 API rejection、部分 chunk 成功和逐字段回读；
- Sheet 同一 record 部分列成功时进入 `INDETERMINATE`，并按实际 cell range 逐格 reconciliation；
- Feishu 测试全部使用 mock，没有真实 API 调用。

### 4.2 数据库与文件 integration contract

~~~bash
pytest tests/integration/test_database_adapter_contracts.py -v
~~~

结果：`2 passed, 2 skipped`。

| Contract | 结果 | 验收解释 |
|---|---|---|
| SQLite temp database | PASS | 真实本地事务、rollback、相同值、missing ID、readback |
| CSV temp file | PASS | 原子替换、磁盘回读、未提供列保留、reconciliation |
| MySQL | SKIP | 未配置 disposable MySQL；不能计为 acceptance |
| PostgreSQL | SKIP | 未配置 disposable PostgreSQL；不能计为 acceptance |

## 5. 最终 Python 门禁

| 命令 | 结果 |
|---|---|
| `pytest tests/ -v -m "not integration"` | PASS：`543 passed, 1 skipped, 11 deselected` |
| `pytest tests/ -v -m integration` | PASS：`9 passed, 2 skipped, 544 deselected` |
| `pytest tests/ --cov=src --cov-branch --cov-report=json:coverage.json` | PASS：`552 passed, 3 skipped`；生成 coverage JSON |
| `python3 .github/scripts/check_coverage.py coverage.json` | PASS |
| `ruff check src/ tests/ cli.py main.py gateway.py .github/scripts` | PASS |
| `black --check src/ tests/ cli.py main.py gateway.py .github/scripts` | PASS：106 files unchanged |
| `mypy src/ --ignore-missing-imports` | PASS：64 source files，无 error |
| `python3 -m py_compile cli.py main.py gateway.py` | PASS |
| `find src -name '*.py' -exec python3 -m py_compile {} \;` | PASS |

本机没有 `python` alias，因此计划中的 `python` 命令使用同一 Python 3.11.9 环境的 `python3` 执行；没有改变测试内容或门槛。

### 5.1 Coverage

| Gate | 实测 | 阈值 | 结果 |
|---|---:|---:|---|
| overall line | 80.08% | 75.00% | PASS |
| overall branch | 66.30% | 65.00% | PASS |
| jobs aggregate line | 92.52% | 85.00% | PASS |
| core-runner aggregate line | 88.96% | 85.00% | PASS |
| gateway aggregate line | 89.71% | 85.00% | PASS |

Coverage 阈值未下调。

### 5.2 全量门禁收敛修正

第一次全量 non-integration 运行发现三组未纳入 H1.0 定向集合的测试仍在构造或索引旧 3.x runtime dict：Gateway protocol YAML、TokenEstimator fixture 和一条旧 `retry_limits` 断言。测试随后统一改为先构造 canonical v4 `RootConfig`，再通过 `compile_job_config()`/Gateway loader 进入现有 runtime；没有在生产代码中增加旧配置 loader、alias 或兼容分支。正式全量复跑通过。

第一次 mypy 运行还发现 processor 的非空 `retry_data` 被标成 Optional，以及 Bitable 局部变量复用妨碍类型收窄。修正仅收紧类型注解和局部变量名，不改变 retry/writeback 行为；正式 mypy 复跑无 error。

## 6. Web 与依赖门禁

| 命令 | 结果 |
|---|---|
| `cd web && npm run lint` | PASS |
| `cd web && npm test` | PASS：4 files / 6 tests |
| `cd web && npm run build` | PASS：60 modules transformed |
| `cd web && npm run test:e2e` | PASS：5 Playwright tests |
| `cd web && npm audit --package-lock-only --audit-level=high` | PASS：0 vulnerabilities |
| `uvx --python <local-python3.11> --from pip-audit pip-audit -r requirements.txt` | PASS：No known vulnerabilities found |
| `actionlint` | PASS |

本机原先没有 `pip-audit` console script。正式依赖审计通过现有 `uvx` 在隔离缓存中运行 `pip-audit`，未修改项目依赖或 lockfile。第一次由 `uvx` 默认 Python 3.12 创建 audit 临时 venv 时，`ensurepip` 收到 `SIGABRT`；改为项目实际使用的本机 Python 3.11 后正式 audit exit 0。CacheControl 输出 cache deserialization warning，但没有改变 audit 结果。

本机 `actionlint 1.7.7` 尚未内置 workflow 使用的 `macos-15-intel` 和 `macos-26` label。`.github/actionlint.yaml` 只将这两个 label 加入识别列表，没有删除或改写 CI matrix。`actionlint` 通过不等于这些 runner label 当前一定可调度，远端 exact-head CI 和平台构建仍属于 H4。

第一次 Playwright 全量运行发现 H1.2 新增的 `unresolved_writes` 未进入 E2E mock，Jobs 页面因此在 mock 数据下抛出运行时错误。补齐 canonical Job counts/status fixture 后，正式复跑 `5 passed`；没有删除测试或降低并行度。

## 7. localhost Control → Gateway source smoke

使用 canonical v4 临时配置、localhost 临时端口和临时测试 token。Gateway 配置了 `example.invalid` 占位 upstream，但 smoke 没有发送 Chat/Responses 请求。

| 检查 | 结果 |
|---|---|
| Control `/health` | 200，`version=4.0.0-dev` |
| Control `/api/status` 无认证 | 401 |
| Control `/api/status` 有认证 | 200 |
| Control 显式 config/port 启动 Gateway | 200，managed status `running` |
| Control 对 Gateway health 回读 | `healthy`，`available_models=1`，`total_models=1` |
| Gateway `/admin/health` 无认证 | 401 |
| Gateway `/admin/health` 有认证 | 200 |
| Gateway `/admin/capabilities` 有认证 | 200 |
| Gateway `/v1/models` 有认证 | 200，1 个本地配置模型 |
| Control 停止 Gateway | 200，managed status `stopped` |
| Control shutdown | application shutdown complete |
| 最终端口/进程检查 | Control/Gateway listener 均为 0，无匹配残留进程 |

第一次调用 Gateway stop 时未发送 JSON `Content-Type`，Control 按合同返回 `415 unsupported_media_type`；正式 stop 使用 `{}` JSON body 后返回 200。这是 smoke 调用前置条件修正，不是进程停止失败。

## 8. Acceptance Ledger 决策

本轮可以离开 `UNREVIEWED` 的范围：

- H1.0 canonical config；
- 3.x config migration/compatibility paths 删除；
- H1.0 version metadata；
- H1.1/H1.2 processor retry、source failure、PreparedResult、receipt、persisted 和终态边界；
- H1.2 datasource base/contracts；
- H1.3 SQLite；
- H1.4 Excel/CSV。

必须继续 `UNREVIEWED` 的范围：

- MySQL disposable real database contract；
- PostgreSQL disposable real database contract；
- Feishu Bitable/Sheet 真实授权环境 UAT；
- H2 Repository/Supervisor/lease/全面恢复矩阵；
- H3 Gateway 完整路由与 affinity 语义；
- H4 Control/Web 产品化、exact-head CI、跨平台打包、Release/部署和业务 UAT。

## 9. Git 与交付边界

H1 source commit 顺序：

1. `d4447c5dee86906bbfc4a2720ca5ed85b3fda234` — `feat!: establish canonical v4 configuration`
2. `cf29b4e013244598e5b1dbc7789cfb0433e24a34` — `fix(core): make task retries source-aware`
3. `eb9ebb5d32471c9b54503b57dec23b8418ce200c` — `feat(core): add prepared-result commit semantics`
4. `f6cba67fd4c8cd20f53a84ca5c49965f59188b22` — `fix(data): harden database commit receipts`
5. `b890799a9485318085217c901223da563730f9ef` — `fix(data): reconcile file and Feishu writeback`

本文件与 Acceptance Ledger 由紧随其后的 `docs: record H1 local acceptance evidence` commit 固化。没有创建远端分支、PR、tag 或 Release，也没有部署。
