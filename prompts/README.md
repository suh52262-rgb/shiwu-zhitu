# Agent 提示词（P13）

这里存放各 Agent 节点的系统提示词正文。**改提示词请改这些 `.md` 文件，不要改 Python 代码。**

## 目录约定

```
prompts/
├── step_matcher/v1.md      学生作答 → Problem Graph 节点匹配
├── analyze_step/v1.md      学生当前进展分析（步骤 / 方法 / 错误 / 知识点 / 是否完成）
├── analyze_step/v2.md      ↑ 加 missing_reasoning / recommended_next_step（P23，当前默认）
├── diagnosis/v1.md         错误归因（类型 / 原因 / 知识点 / 严重程度 / 建议）
├── tutor/v1.md             按 hint_level 生成分级提示
├── graph_generator/v1.md   P14：由题面 + 参考答案生成候选解题图
└── graph_validator/v1.md   P15：审校候选图的物理一致性
```

**同一角色可以并存多个版本**，`DEFAULT_VERSIONS` 决定用哪个，Trace 每轮记录实际用了哪个。
`analyze_step` 的 v1/v2 就是一次真实的多版本并存：v1 是 P23 之前的版本，
保留它是为了能对照"加字段前后行为有没有变化"。

文件内容即提示词原文，注册表不做任何裁剪或拼接。结构化输出（JSON 格式要求）由
`services.llm.chat_structured` 自动追加，**不需要写进提示词**。

## 职责边界（任务书 P13 强制要求）

| 角色 | 只负责 | 明确不负责 |
|---|---|---|
| `step_matcher` | 输入对应哪些节点、是否覆盖多节点（跳步）、是否正确、是否缺关键推理、是否走了合理替代路线 | **生成教学提示** |
| `analyze_step` | 描述学生当前进展 | 错误归因、生成提示 |
| `diagnosis` | 错误类型 / 原因 / 知识点 / 严重程度 / 是否需干预 | 判断学生做到哪一步、生成提示 |
| `tutor` | 按给定 `hint_level` 与诊断结果生成提示 | **重新判断学生做到了哪一步** |
| `graph_generator` | 由题面与参考答案整理出轻量的解题步骤图（节点 / 边 / 路径 / 典型错误 / 知识点） | 教学提示、判断学生进展；也**不做物理正确性的最终裁决**（那是 P15 校验器的事） |
| `graph_validator` | 审校一张解题图在**物理上**是否站得住（规律误用 / 条件缺失 / 系统边界 / 与题目不符 / 结论矛盾） | 结构合法性、流程完整性、路径一致性（这三类由规则做，不该问 LLM）；也不批改单位与化简笔误 |

> `graph_validator` 的提示词里有一条硬要求：**没发现问题就返回 `consistent=true`**，
> 为了"显得在干活"而凑数报问题，比漏报更有害。

## 加一个新版本

1. 新建 `prompts/<角色>/v2.md`（内容基于上一版修改）；
2. 把 `backend/app/agent/prompt_registry.py` 的 `DEFAULT_VERSIONS[<角色>]` 改为 `"v2"`。

也可以不改代码，用环境变量临时覆盖（方便 A/B 与线上排查）：

```
PROMPT_STEP_MATCHER_VERSION=v2
PROMPT_TUTOR_VERSION=v2
```

## 回归要求

提示词属于"改了就必须重新验证"的东西。改完至少跑：

```bash
.venv/Scripts/python -m pytest tests -q                 # 单测
.venv/Scripts/python scripts/run_simulation.py --all --offline   # 五类学生剧本（离线）
.venv/Scripts/python scripts/run_simulation.py --all             # 五类学生剧本（真实模型）
.venv/Scripts/python scripts/run_eval.py --suite all             # 五类评测
```

## 版本如何被记录

每一轮分析都会把当轮实际使用的版本写进 `analysis_traces.payload`，形如：

```json
{
  "step_matcher_prompt_version": "v1",
  "analyze_step_prompt_version": "v2",
  "diagnosis_prompt_version": "v1",
  "tutor_prompt_version": "v1",
  "graph_generator_prompt_version": "v1",
  "graph_validator_prompt_version": "v1"
}
```

这样出现异常输出时，可以先确认"是提示词版本换了"还是"模型/数据的问题"。
（`graph_generator` / `graph_validator` 的版本也随每轮 Trace 记录，虽然它们只在
建图与校验时才真正用到——统一记录的好处是"当前进程用的是哪一版"永远只有一个答案。）
