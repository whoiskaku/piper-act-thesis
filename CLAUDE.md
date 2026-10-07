# 全局准则

## GitHub 认证方针

- 需要 Git/GitHub 操作（push、pull、gh 命令等）时，提示用户执行 `gh auth login`
- 操作完成后执行 `gh auth logout`
- 不要让 token 保持登录状态

## Git 运用方针

- 在 develop 分支上进行开发和提交
- 在合适的时机通过 `git merge --no-ff develop` 合并到 main 分支
- 不受 Git 管理的文件不在项目处理范围内（但工作中新建的文件属于处理对象）
- 提交信息中不添加 `Co-Authored-By`

## 工具使用方针

- 文件搜索使用 `Glob` 工具，而非 `find` 命令
- 文本搜索使用 `Grep` 工具，而非 `grep`/`rg` 命令
- 文件读取使用 `Read` 工具，而非 `cat`/`head`/`tail`

## 计划模式运用

- 在调用 ExitPlanMode 之前，先判断执行本次计划时「clear context and auto-accept edits」和「auto-accept edits」哪个更合适，以文本形式明确推荐后再调用 ExitPlanMode

## 老师要求（每次行动的标准）

- 每次行动（计划、实验、代码、汇报）都必须逐条回答 `lerobot-piper/reports/ADVISOR_STANDARD.md` 的第 1–6 条
- 开头先写「本次回答第几条、怎么回答」；与本次无关的条写「本次不涉及」并说明理由
- 答不上来的条目，先补齐证据再行动
- 第 7–9 条由用户自行处理，不在标准内


