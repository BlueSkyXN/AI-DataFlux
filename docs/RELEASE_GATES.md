# 4.0 发布门禁与 UAT 边界

本页定义“代码已完成”到“用户可用”之间的证据层。当前活动源码版本号是 `4.0.0-dev`；本地版本切换不表示已创建 tag、发布 Release 或部署。

2026-09-07 工作区的最新验证与打包阻塞见 [V4_LOCAL_VALIDATION.md](./V4_LOCAL_VALIDATION.md)。下文 2026-08-31 数字为历史快照，不能替代当前实现的验证。

## 证据层

| 层级 | 必需证据 | 不能替代的下一层 |
|---|---|---|
| 实现 | 源码与聚焦测试存在 | 全量回归 |
| 本地验证 | pytest、coverage、Ruff、Black、mypy、前端 lint/test/build、dependency audit | CI |
| CI | 目标 commit 的所有 required checks 绿色 | 打包产物 |
| 打包 | 每个平台/变体的二进制生成且 smoke 通过 | GitHub Release |
| Release | tag 、commit 和 artifacts 回读一致 | 部署 |
| 部署 | 在线版本与预期 commit 一致，健康端点可读 | 业务 UAT |
| UAT | 真实用户流程、数据写回、取消/恢复、权限与失败路径验收 | — |

## CI 门禁

`.github/workflows/test.yml` 应阻断：

- 固定版本的 Ruff 和 Black；
- 无 `|| true` 的 mypy typed baseline；
- Python 3.10–3.14 多平台非 integration 测试；
- 包含 `src/gateway/` 的 coverage：总体 line ≥75%、branch ≥65%；`src/jobs/`、`src/core/processor.py + src/core/job_runner.py`、`src/gateway/` 三个关键集合聚合 line ≥85%，且集合中任一单文件 line 不得低于 70%；
- `pip-audit -r requirements.txt`；
- 前端 `npm ci`、`npm run lint`、`npm run test`、`npm run test:e2e`、`npm run build` 和 `npm audit --audit-level=high`；
- CLI 帮助/配置验证与本地 integration tests。

前端已经定义基于 Vitest 的 `test` script；CI 和打包 workflow 都无条件执行 `npm run test`，脚本缺失或测试失败均会阻断。

## 当前本地门禁快照（2026-08-31）

该快照用于暴露当前阻塞，不替代目标 commit 的 CI：

- non-integration pytest：502 passed、1 skipped、9 deselected；本地 integration：7 passed、2 个真实数据库 contract 因未启用 disposable service 而 skipped；
- coverage gate：通过。overall line 79.56%、overall branch 66.18%；
- `src/jobs` aggregate line 92.79%、core runner aggregate line 88.25%、Gateway aggregate line 90.01%，关键集合均达到 85%；
- 固定版本 Ruff `0.16.4`、Black `25.11.0` 与 mypy `1.20.2`：通过；mypy 对 `src/` 的 62 个 source files 无错误；
- 前端 ESLint、Vitest（4 files、5 tests）、production build、Playwright E2E（5 tests）：通过；
- `npm audit --audit-level=high`：0 vulnerability；`pip-audit -r requirements.txt`：No known vulnerabilities found；
- 本地源码 Control `/health`、Control API 401/授权访问、Control 启停 Gateway 并携 Bearer 回读 health，以及 Gateway `/admin/health`、`/admin/capabilities`、`/v1/models` 已 smoke 通过。

上述阈值不得为获得绿色结果而下调。覆盖率测试、前端依赖升级和 lockfile 变更应作为独立实现工作处理并重新验证。

## 打包 smoke

PyInstaller 的 Full/CLI-only 变体和 Nuitka Full 变体在上传 artifact 前必须从生成的二进制执行：

```text
version
--help
check
process --config config-example.yaml --validate
```

Full 变体还必须通过 `gui --help`，实际启动 Control 并在 60 秒内回读 `/health`；CLI-only 变体反而必须确认 `gui` 不可用。smoke 失败时不上传 artifact。

这些二进制 smoke 只能在打包 matrix 产生实际可执行文件后验证，本地未执行。`actionlint` 在忽略 runner-label 数据库告警时通过；其内置 label 列表尚不认识 `macos-15-intel` 和 `macos-26`，在确认 GitHub 当前 runner 可用性前应视为 CI 调度风险，不能静默删除平台矩阵。

## Release 前核对

1. `src.__version__`、tag 和 Release 名称一致；开发版不使用稳定 tag。
2. 8 个 PyInstaller artifact 与 4 个 Nuitka artifact（合计 12 个）均存在，文件名与 Release 文案一致。
3. Full 与 CLI-only 的 `gui` 边界与实际二进制一致。
4. migration、Gateway capability、Control/Job API 和 CLI JSON schema 已冻结。
5. 已记录已知问题和回退方法。回退是恢复前一个 artifact/配置备份，不是就地修改 Job 持久化文件。

## 业务 UAT

至少逐项验收：

- workspace root 内配置可提交，绝对路径、`..` 和 symlink 越界被拒绝；
- submit/list/get/event/cancel/resume 从 Control API 和 CLI JSON 视角返回一致 schema；
- Job 从 `queued` 进入终态，`counts.persisted/failed` 与数据源真实写回一致；
- Control/Gateway 统一 token 的正确、缺失、错误和非 loopback 启动路径；
- Chat/Responses 的非流式与 SSE，以及 tools、multimodal、`n>1`、logprobs、json_schema、`previous_response_id`；
- 中断后 lease/recovery，config 变化时进入 `blocked`；
- 真实 Feishu/数据库写回仅在专用测试数据和明确授权下执行。

只有上述业务流程有可回读证据时，才能声明 4.0 已完成 UAT。
