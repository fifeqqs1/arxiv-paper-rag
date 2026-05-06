# Feishu RAG 轻量量化评估说明

本文档记录当前 `arxiv_rag` 项目里 Feishu RAG 线路的轻量评估方案：如何构造测试集、如何调用评估接口、评估哪些指标、为什么选择这些指标、如何运行 Ragas、如何解读报告，以及如何对比旧版和新版性能。

这套方案的定位不是做大规模线上 A/B，也不是做严格学术 benchmark，而是给日常开发提供一个可重复、低成本、能快速暴露问题的质量回归流程。它主要回答这些问题：

- Feishu 入口是否能根据用户问题正确走到论文推荐、论文追问、对比、重置等路径。
- 回答是否被检索到的上下文支撑，是否出现明显幻觉。
- 回答是否真正回应了用户的问题，而不是泛泛介绍论文。
- 检索出来的 contexts 是否有用，排序是否合理。
- 会话记忆是否能支持“第一篇”“这篇”“这几篇”等追问。
- 重置会话后，系统是否还错误引用上一轮论文。
- 修改 intent、prompt、retrieval、reranker 或 memory 之后，质量是否退化。

## 目录与核心文件

当前评估相关文件如下：

```text
eval/
  feishu_ragas_smoke.jsonl          # 通用轻量样本
  feishu_uav_ragas_smoke.jsonl      # 当前主要使用的 UAV 领域 15 条样本
  reports/
    feishu_ragas_latest.md          # 最近一次评估的 Markdown 摘要，会被覆盖
    feishu_ragas_<timestamp>.json   # 每次评估的完整 JSON 报告，不会自动覆盖

scripts/
  eval_feishu_ragas.py              # 评估入口脚本

src/routers/feishu_local.py         # 本地 Feishu reply API，用于非飞书客户端和评估
src/schemas/api/feishu.py           # /api/v1/feishu/reply 请求/响应 schema
src/schemas/api/ask.py              # /ask 的 include_contexts 评估开关
```

主要运行脚本是：

```bash
uv run python scripts/eval_feishu_ragas.py --dataset eval/feishu_uav_ragas_smoke.jsonl
```

## 评估不影响正常运行

评估能力通过显式开关启用，默认不影响正常 Feishu/RAG 使用。

`/api/v1/feishu/reply` 请求里新增了：

```json
{
  "session_id": "some_session",
  "query": "用户问题",
  "eval_debug": true
}
```

关键点：

- `eval_debug` 默认是 `false`。
- 正常 Feishu 使用不需要传 `eval_debug`，响应仍然只关心最终答案。
- 只有 `eval_debug=true` 时，接口才额外返回 `contexts`、`sources`、`intent`、`rewritten_query`、`route` 等调试字段。
- `/ask` 的 `include_contexts` 默认也是 `false`，只有评估链路需要 contexts 时才打开。
- 评估脚本只通过 HTTP 调用 API，不直接修改索引、论文库或核心检索逻辑。
- 评估脚本会自动给 session 加时间戳前缀，避免和真实用户 session 混用。
- 评估会消耗 LLM 和 embedding API 调用额度，也可能写入隔离后的 eval 会话记录；它不是线上用户流量。

因此，评估相关字段属于“只读观测和离线打分能力”，不是正常运行路径的业务依赖。

## 整体评估流程

一次完整评估的流程如下：

```text
JSONL 测试集
    |
    v
scripts/eval_feishu_ragas.py
    |
    | 逐条 POST /api/v1/feishu/reply, eval_debug=true
    v
Feishu RAG 线路
    |
    | 返回 answer + contexts + sources + intent + rewritten_query + route
    v
规则检查
    |
    | expected_contains / expected_not_contains / source / memory / latency
    v
Ragas 评分
    |
    | Faithfulness / ResponseRelevancy / LLMContextPrecisionWithoutReference
    v
报告输出
    |
    +-- eval/reports/feishu_ragas_latest.md
    +-- eval/reports/feishu_ragas_<timestamp>.json
```

评估分两层：

1. 规则检查：便宜、稳定、可解释，适合判断硬性要求。
2. Ragas 检查：使用 LLM judge 和 embedding judge，适合判断“答案是否有支撑”“是否回答问题”“检索片段是否有用”。

这两层互补。规则检查能抓住 reset 后还引用论文、缺少来源、关键词没有覆盖等明显问题；Ragas 能抓住答案虽然包含关键词但实际没有被上下文支撑、或者回答偏题的问题。

## 测试集设计

当前主要测试集是：

```text
eval/feishu_uav_ragas_smoke.jsonl
```

它是 UAV 领域的 15 条轻量样本，覆盖：

- 论文推荐
- 单篇论文追问
- 多篇对比
- 实验/方法/传感器/局限追问
- 会话记忆
- 会话重置

每行是一个 JSON 对象，字段如下：

| 字段 | 是否必需 | 作用 |
| --- | --- | --- |
| `case_id` | 是 | 样本唯一 ID，报告里用它定位问题 |
| `session_id` | 是 | 会话 ID，同一组多轮问答必须使用同一个 session |
| `type` | 是 | 样本类型，例如 `uav_paper_search`、`uav_memory_method`、`reset` |
| `query` | 是 | 用户问题 |
| `expected_contains` | 否 | 答案中必须出现的关键词；缺失则规则失败 |
| `expected_not_contains` | 否 | 答案中禁止出现的词；命中则规则失败 |
| `requires_source` | 否 | 是否要求有来源；默认大多数问题都要求来源 |
| `memory_case` | 否 | 是否计入会话记忆通过率 |
| `reference_answer` | 否 | 参考答案说明；第一版主要用于人工理解和后续扩展 |

一个推荐样本示例：

```json
{
  "case_id": "uav_search_001",
  "session_id": "uav_eval_nav_001",
  "type": "uav_paper_search",
  "query": "给我推荐三篇无人机视觉导航或者自主探索相关论文",
  "expected_contains": ["arXiv", "论文"],
  "expected_not_contains": [],
  "requires_source": true,
  "reference_answer": "应推荐当前索引中与 UAV 视觉导航、自主探索或无人机感知相关的真实 arXiv 论文。"
}
```

一个追问样本示例：

```json
{
  "case_id": "uav_followup_001",
  "session_id": "uav_eval_nav_001",
  "type": "uav_memory_method",
  "query": "第一篇用了什么核心方法？",
  "expected_contains": ["方法"],
  "expected_not_contains": ["我现在还没有记住"],
  "requires_source": true,
  "memory_case": true,
  "reference_answer": "应基于上一轮第一篇 UAV 相关论文，说明其核心方法、输入信息和关键模块。"
}
```

一个重置后检查样本示例：

```json
{
  "case_id": "uav_memory_after_reset_001",
  "session_id": "uav_eval_slam_001",
  "type": "memory_after_reset",
  "query": "第一篇用了什么方法？",
  "expected_contains": ["还没有记住"],
  "expected_not_contains": ["参考来源", "arXiv"],
  "requires_source": false,
  "memory_case": true,
  "reference_answer": "重置后不应继续引用上一轮论文。"
}
```

### 为什么样本数量先用 15 条

目前目标是轻量回归，不是完整 benchmark。15 条的好处是：

- 跑得动，适合每次改 Feishu prompt、intent、retrieval 后快速检查。
- 能覆盖最核心的链路：推荐、追问、对比、重置。
- 样本少，失败后人工复核成本低。
- 对 Qwen API 调用压力较小，减少限流和超时概率。

如果后续要做更正式的版本对比，可以扩展到 30-50 条，并把不同任务类型保持固定比例。

## 评估接口与调试字段

评估脚本调用：

```http
POST /api/v1/feishu/reply
```

请求：

```json
{
  "session_id": "ragas_20260422_222247_uav_eval_nav_001",
  "query": "第一篇用了什么核心方法？",
  "eval_debug": true
}
```

响应里评估需要的字段：

| 字段 | 作用 |
| --- | --- |
| `answer` | 模型最终回答，是规则和 Ragas 的主要被评对象 |
| `contexts` | 检索到并喂给模型的上下文片段，Ragas 用它判断支撑性和检索质量 |
| `sources` | 论文来源 URL，规则检查用它计算 source rate |
| `intent` | Feishu intent 分类结果，用于排查问题是否走错路由 |
| `rewritten_query` | Feishu 改写后发给 RAG 的问题，用于排查追问改写是否失败 |
| `route` | 实际路由，例如 paper search、paper-focused ask、general RAG |
| `latency_ms` | 评估脚本记录的端到端延迟 |

`contexts` 是 Ragas 必需的。没有 contexts 的样本无法计算 Faithfulness、ResponseRelevancy、ContextPrecision 等 Ragas 指标，但仍可以做规则检查。

## 使用的工具

### 1. Ragas

Ragas 用于 RAG 质量评估。本项目目前只使用三个核心指标：

- `Faithfulness`
- `ResponseRelevancy`
- `LLMContextPrecisionWithoutReference`

没有一开始引入更多指标，是为了保持第一版简单、可解释、运行成本可控。

### 2. datasets

Ragas 接收 Hugging Face `datasets.Dataset` 格式的数据。脚本会把每条样本整理成：

```python
{
    "user_input": query,
    "response": answer,
    "retrieved_contexts": contexts,
    "reference": reference_answer,
}
```

### 3. langchain-openai

用于通过 OpenAI-compatible 接口接入 Qwen judge model 和 embedding model。

### 4. Qwen / DashScope OpenAI-compatible API

当前评估 judge 默认配置：

| 项 | 默认值 |
| --- | --- |
| Chat judge model | `qwen3.5-plus` |
| Embedding model | `text-embedding-v4` |
| Base URL | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| API key env | `QWEN_API_KEY` |
| Thinking | `enable_thinking=false` |

使用 Qwen 的原因：

- 项目本身已经在 `.env` 中维护 `QWEN_API_KEY`，不需要额外引入另一套 key。
- OpenAI-compatible 接口可直接被 `langchain-openai` 使用。
- 中文问题、中文回答、论文领域内容的 judge 体验更贴近当前应用。
- embedding model 也可以复用同一套 DashScope 接口。

### 5. httpx

评估脚本使用 `httpx.Client` 顺序调用 `/api/v1/feishu/reply`，并记录每条样本的端到端延迟。

脚本对临时错误做了重试：

- `429`
- `500`
- `502`
- `503`
- `504`
- 网络请求异常

默认重试参数：

```text
--request-retries 3
--retry-backoff 20
```

也可以在 Qwen 容量波动时调大：

```bash
--request-retries 5 --retry-backoff 30
```

## 指标说明与选择理由

### 1. Faithfulness

含义：答案中的陈述是否能被 retrieved contexts 支撑。

为什么选它：

- RAG 的核心要求是“基于检索上下文回答”。
- 论文问答很容易出现模型补全、猜测实验细节、虚构 baseline 等问题。
- Faithfulness 可以帮助发现“答案看起来专业，但上下文并没有支持”的情况。

典型问题：

- 答案提到了某个实验指标，但 contexts 里没有。
- 答案说论文用了某种方法，但检索片段只说了任务背景。
- 答案把另一篇论文的信息混到了当前论文里。

当前阈值：

```text
faithfulness >= 0.80
```

低于 0.80 时，通常要人工检查失败样本的 answer 和 contexts，判断是检索没召回、prompt 没约束住，还是 Ragas judge 误判。

### 2. ResponseRelevancy

含义：答案是否回应了用户问题。

为什么选它：

- Feishu 多轮问答里，用户经常问“第一篇用了什么方法”“这几篇有什么局限”。
- 系统可能检索到了相关论文，但回答成了论文摘要，没有回答“方法/实验/局限/传感器输入”等具体点。
- ResponseRelevancy 可以发现答非所问、回答过泛、没有抓住追问意图的问题。

典型问题：

- 用户问“实验怎么验证”，答案只介绍论文背景。
- 用户问“用了哪些传感器”，答案只说任务是无人机导航。
- 用户问“几篇有什么区别”，答案只单篇总结。

当前阈值：

```text
response_relevancy >= 0.75
```

这个指标目前是两版都偏低的主要短板，说明回答聚焦度还有提升空间。

### 3. LLMContextPrecisionWithoutReference

含义：不依赖标准答案，由 LLM 判断 retrieved contexts 是否对回答问题有用，以及前排上下文是否更相关。

为什么选它：

- 当前测试集没有完整人工 reference answer。
- 论文 RAG 的检索质量不只看是否召回，还要看前几个片段是否对问题有用。
- 对 Feishu 追问来说，如果上下文排序错了，模型很容易回答上一轮错误论文或错误片段。

典型问题：

- contexts 里有 UAV 论文，但和用户问的实验/传感器/局限无关。
- 第一篇、第二篇追问时，检索到了别的论文片段。
- 推荐型问题返回的摘要片段太泛，不能支撑后续追问。

当前阈值：

```text
context_precision >= 0.65
```

### 4. rule pass rate

含义：简单规则检查的整体通过率。

规则包括：

- `expected_contains` 中的词必须出现在答案里。
- `expected_not_contains` 中的词不能出现在答案里。
- 需要来源的问题必须有 `sources`。

为什么选它：

- 规则检查稳定，不依赖 LLM judge。
- 对 reset、来源、会话记忆这类硬约束，规则比 Ragas 更直接。
- 可以快速发现明显回归。

当前阈值：

```text
pass_rate >= 80%
```

### 5. source rate

含义：所有 `requires_source=true` 的样本中，有来源的比例。

为什么选它：

- 论文 RAG 需要让用户知道答案来自哪些论文。
- Feishu 推荐和论文追问没有来源时，可验证性明显下降。
- source rate 是非常直接的产品质量指标。

当前阈值：

```text
source_rate >= 90%
```

### 6. memory pass rate

含义：会话记忆类样本中的规则通过率。

为什么选它：

- Feishu RAG 的关键体验是多轮论文追问。
- “第一篇”“这篇”“这两篇”的解析依赖 conversation memory。
- reset 后不应继续引用旧上下文。

当前阈值：

```text
memory_pass_rate >= 80%
```

### 7. latency p50 / p95

含义：评估脚本记录的端到端接口耗时。

为什么记录：

- RAG 质量提升不能完全牺牲可用性。
- 追问类样本可能需要检索、改写、生成，延迟比推荐型高很多。
- p95 能暴露少数样本极慢、Qwen 超时、重试等问题。

当前没有设置硬性 latency gate，因为轻量评估受外部 API 波动影响较大。但做版本对比时需要观察 p50/p95 是否明显恶化。

## 当前通过标准

当前轻量评估的通过标准写在 `scripts/eval_feishu_ragas.py` 的 `PASS_THRESHOLDS` 中：

```python
PASS_THRESHOLDS = {
    "faithfulness": 0.80,
    "response_relevancy": 0.75,
    "context_precision": 0.65,
    "pass_rate": 0.80,
    "source_rate": 0.90,
    "memory_pass_rate": 0.80,
}
```

解释：

- `faithfulness >= 0.80`：答案必须主要由 contexts 支撑。
- `response_relevancy >= 0.75`：答案要回应问题，而不是泛泛总结。
- `context_precision >= 0.65`：检索片段至少要大体有用。
- `pass_rate >= 80%`：允许少数轻量样本失败，但不能系统性失败。
- `source_rate >= 90%`：论文问答绝大多数情况下必须有来源。
- `memory_pass_rate >= 80%`：会话追问和 reset 不能大面积失效。

注意：Ragas 分数不是绝对真理。只要指标低于阈值，就应该抽取失败样本人工看 answer、contexts、sources 和 rewritten_query，确认是真问题还是 judge 误判。

## 如何运行评估

### 1. 准备环境变量

确保 `.env` 中有：

```bash
QWEN_API_KEY=...
```

可选覆盖项：

```bash
RAGAS_JUDGE_MODEL=qwen3.5-plus
RAGAS_EMBEDDING_MODEL=text-embedding-v4
QWEN_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
FEISHU_EVAL_BASE_URL=http://localhost:8000
```

脚本会读取 `.env`，但不会覆盖已经 export 的环境变量。

### 2. 启动 API 服务

常见本地启动方式：

```bash
uv run uvicorn src.main:app --host 127.0.0.1 --port 8000
```

如果已经通过 Docker/compose 或其他方式启动，只要能访问对应 base URL 即可。

建议先检查健康状态：

```bash
curl -sS http://127.0.0.1:8000/api/v1/health
```

至少应确认：

- database 可用
- OpenSearch 可用
- LLM/Qwen 可用
- `arxiv-papers-chunks` 索引中有文档

### 3. 先跑 2 条 API 冒烟

正式评估前，建议先取前 2 条样本跑 API 冒烟，不跑 Ragas：

```bash
head -n 2 eval/feishu_uav_ragas_smoke.jsonl > /tmp/feishu_uav_smoke_2.jsonl

uv run python scripts/eval_feishu_ragas.py \
  --dataset /tmp/feishu_uav_smoke_2.jsonl \
  --base-url http://127.0.0.1:8000 \
  --timeout 420 \
  --skip-ragas \
  --request-retries 1 \
  --retry-backoff 10
```

这个步骤主要确认：

- `/api/v1/feishu/reply` 能返回 200。
- `eval_debug=true` 能返回 contexts。
- sources 不为空。
- 会话追问能接上上一轮。
- Qwen key 没有缺失。

如果 2 条冒烟都不通过，不要直接跑完整评估，应先修接口或环境。

### 4. 跑完整 UAV 15 条评估

```bash
uv run python scripts/eval_feishu_ragas.py \
  --dataset eval/feishu_uav_ragas_smoke.jsonl \
  --base-url http://127.0.0.1:8000 \
  --timeout 420 \
  --request-retries 5 \
  --retry-backoff 30
```

运行时会逐条输出进度：

```text
[1/15] Running uav_search_001
[2/15] Running uav_followup_001
...
```

接口调用完成后会进入 Ragas 阶段，看到类似：

```text
Evaluating:  33%|███▎      | 13/39 [...]
```

这里的 39 不是样本数，而是 Ragas 内部根据指标拆出来的评分任务数量。

### 5. 只跑规则检查

如果只是快速检查 Feishu 链路，不想消耗 Ragas judge 调用：

```bash
uv run python scripts/eval_feishu_ragas.py \
  --dataset eval/feishu_uav_ragas_smoke.jsonl \
  --base-url http://127.0.0.1:8000 \
  --skip-ragas
```

此时 Ragas 分数会是 `None`，但规则通过率、来源、记忆、延迟仍然会输出。

## 输出报告

每次完整运行会输出：

```text
eval/reports/feishu_ragas_<timestamp>.json
eval/reports/feishu_ragas_latest.md
```

### Markdown 报告

`feishu_ragas_latest.md` 是最近一次结果摘要，适合快速看：

- 总样本数
- 规则通过率
- source rate
- memory pass rate
- latency p50/p95
- 三个 Ragas 均值
- 每条样本是否通过
- 每条样本的 Ragas 分数和失败原因

注意：`latest.md` 每次都会覆盖。

### JSON 报告

`feishu_ragas_<timestamp>.json` 是完整报告，适合做版本对比和后续分析。

结构：

```json
{
  "summary": {
    "total": 15,
    "pass_rate": 0.8667,
    "source_rate": 1.0,
    "memory_pass_rate": 0.8,
    "latency_p50_ms": 67180.42,
    "latency_p95_ms": 479820.64,
    "ragas": {
      "faithfulness": 0.6453,
      "response_relevancy": 0.4626,
      "context_precision": 0.8192
    },
    "thresholds": {}
  },
  "results": [
    {
      "case_id": "uav_followup_001",
      "type": "uav_memory_method",
      "query": "第一篇用了什么核心方法？",
      "answer": "...",
      "contexts": ["..."],
      "sources": ["..."],
      "intent": "...",
      "rewritten_query": "...",
      "route": "...",
      "latency_ms": 103128.42,
      "rule_pass": true,
      "missing_terms": [],
      "forbidden_hits": [],
      "ragas": {
        "faithfulness": 0.9285,
        "response_relevancy": 0.3895,
        "context_precision": 1.0
      }
    }
  ]
}
```

排查问题时优先看 JSON，因为里面保留了 answer、contexts、sources 和路由信息。

## 如何解读失败样本

建议按这个顺序看：

1. 看 `rule_pass` 是否为 false。
2. 如果 `missing_terms` 非空，说明答案没有覆盖硬性关键词。
3. 如果 `forbidden_hits` 非空，说明答案出现了禁词，reset 后引用旧论文通常属于这一类。
4. 如果 `source_ok=false`，说明需要来源的问题没有 sources。
5. 看 `intent` 和 `route`，判断是不是路由错了。
6. 看 `rewritten_query`，判断追问是否被正确改写。
7. 看 `contexts`，判断检索是否召回了正确论文和正确片段。
8. 看 `answer`，判断模型是否没有遵循 contexts，或者没有回应问题。
9. 最后结合 Ragas 分数判断问题类型。

常见失败类型：

| 现象 | 可能原因 | 处理方向 |
| --- | --- | --- |
| 缺少 `实验`、`方法`、`局限` 等关键词 | 回答没有聚焦用户追问 | 强化 intent/prompt，让回答必须围绕问题维度 |
| reset 后还有 `arXiv` 或参考来源 | 会话清理不彻底或 fallback 策略不对 | 检查 memory reset 和无上下文追问策略 |
| source rate 下降 | Feishu answer 格式没有带 sources | 检查 source 提取和响应组装 |
| Faithfulness 低 | 答案没有被 contexts 支撑 | 改检索、改 prompt，要求不知道就说明片段未覆盖 |
| ResponseRelevancy 低 | 答非所问或回答太泛 | 改 query rewrite、intent 分类、回答模板 |
| ContextPrecision 低 | 检索片段无关或排序差 | 改 retrieval query、reranker、paper-focused retrieval |
| 延迟 p95 很高 | Qwen 超时、重试、某类问题上下文太大 | 降低上下文、增加超时保护、检查慢样本 |

## 旧版和新版如何公平对比

对比不同版本时，要保证以下条件一致：

- 使用同一份 JSONL 测试集。
- 使用同一个 OpenSearch 索引和论文数据。
- 使用同一个 judge model 和 embedding model。
- 使用同一套阈值。
- 使用相同或接近的 API timeout/retry 参数。
- 不要把一个版本的 `latest.md` 和另一个版本不同时间的非同源数据混在一起比较。

推荐流程：

1. 在新版当前代码上跑完整评估，保留 timestamp JSON。
2. 用 `git worktree` 拉出旧版 commit。
3. 如果旧版没有 `/api/v1/feishu/reply` 或 `eval_debug`，只在临时 worktree 中加一个评估适配层。
4. 评估适配层只能暴露旧版原有 Feishu/RAG 逻辑的 answer、contexts、sources，不能修改旧版核心检索、prompt、reranker、memory 行为。
5. 启动旧版服务到另一个端口，例如 `8011`。
6. 用同一份测试集和同一个 `scripts/eval_feishu_ragas.py` 跑旧版。
7. 对比两个 timestamp JSON 的 `summary` 和失败样本。

示例命令：

```bash
git worktree add /tmp/arxiv_rag_b88aa15 b88aa15

# 在 /tmp/arxiv_rag_b88aa15 启动旧版服务，端口用 8011
# 如果旧版缺少 eval_debug，只在该临时 worktree 加 eval-only adapter

uv run python scripts/eval_feishu_ragas.py \
  --dataset eval/feishu_uav_ragas_smoke.jsonl \
  --base-url http://127.0.0.1:8011 \
  --timeout 420 \
  --request-retries 5 \
  --retry-backoff 30
```

注意：旧版如果本身没有 contexts 输出，就必须加临时 adapter，否则 Ragas 无法判断 Faithfulness 和 ContextPrecision。这个 adapter 的边界必须很清楚：只加观测字段，不改业务逻辑。

## 当前一次旧新版对比记录

以下是一次已完成的 UAV 15 条评估结果，方便作为历史参考。

旧版：

```text
eval/reports/feishu_ragas_20260422_222247.json
```

新版：

```text
eval/reports/feishu_ragas_20260422_173017.json
```

对比：

| 指标 | 旧版 b88aa15 | 新版 | 结论 |
| --- | ---: | ---: | --- |
| rule pass rate | 86.7% | 80.0% | 旧版略好 |
| source rate | 100.0% | 100.0% | 持平 |
| memory pass rate | 80.0% | 70.0% | 旧版略好 |
| faithfulness | 0.645 | 0.670 | 新版略好 |
| response relevancy | 0.463 | 0.494 | 新版略好 |
| context precision | 0.819 | 0.765 | 旧版略好 |
| latency p50 | 67.2s | 76.9s | 旧版略快 |
| latency p95 | 479.8s | 148.8s | 新版明显更稳 |

这次结果的结论：

- 新版在 Faithfulness 和 ResponseRelevancy 上略有提升。
- 旧版在规则通过率、memory pass rate、context precision 上略好。
- 两版 source rate 都是 100%。
- 两版都没有通过完整门槛，主要短板仍是 Faithfulness 和 ResponseRelevancy。
- 15 条样本规模较小，只能作为轻量回归信号，不能当作绝对性能结论。

旧版失败样本：

- `uav_followup_002`：缺少“实验”。
- `uav_memory_after_reset_001`：reset 后仍出现 `arXiv`。

新版失败样本：

- `uav_followup_002`：缺少“实验”。
- `uav_followup_007`：缺少“局限”。
- `uav_memory_after_reset_001`：reset 后仍出现 `arXiv`。

## 什么时候应该跑评估

建议在这些改动后跑：

- 改 Feishu intent 分类。
- 改 Feishu prompt。
- 改 query rewrite。
- 改论文推荐逻辑。
- 改 paper-focused retrieval。
- 改 OpenSearch query builder。
- 改 reranker。
- 改 context 拼接策略。
- 改 conversation memory。
- 改 reset 逻辑。
- 改 source 提取或回答格式。

建议节奏：

1. 小改动：先跑 2 条 smoke，再跑 `--skip-ragas` 全量规则。
2. prompt/retrieval/memory 改动：跑完整 15 条 Ragas。
3. 准备合入重要改动：和上一个稳定报告做 JSON 对比。

## 如何扩展测试集

扩展时不要只堆推荐型问题，应保持任务结构平衡。

建议比例：

- 30%-40%：论文推荐
- 30%-40%：单篇论文追问
- 10%-20%：多篇对比
- 10%-20%：会话记忆和重置

新增样本时注意：

- `session_id` 要能表达多轮关系。
- 同一组追问必须排在推荐样本之后。
- `expected_contains` 不要写太苛刻，优先写任务维度词，例如“实验”“方法”“传感器”“局限”。
- `expected_not_contains` 只放明确不该出现的内容，例如 reset 后的 `arXiv`、`参考来源`。
- 推荐型问题应设置 `requires_source=true`。
- reset 和 reset 后无上下文追问一般设置 `requires_source=false`。
- `reference_answer` 可以先写“应该回答什么维度”，不必写成完整标准答案。

## 局限性

这套评估有几个明确局限：

- 样本只有 15 条，统计显著性不足。
- Ragas 是 LLM judge，结果会受 judge model、prompt、API 状态影响。
- Ragas 分数不是绝对事实，失败样本必须人工复核。
- `reference_answer` 当前不作为强监督标准答案使用。
- latency 包含外部 API 波动，p95 尤其容易受 Qwen 超时或限流影响。
- `ResponseRelevancy` 对中文、论文术语、长回答有时会偏严。
- `context_precision` 可能出现 `nan`，通常是该样本的上下文或 judge 任务无法稳定评分。
- 推荐型问题的 Faithfulness 可能偏低，因为推荐回答常包含标题、链接、摘要混合信息。
- 如果旧版通过临时 adapter 输出 contexts，必须确认 adapter 没有改变旧版核心行为。

## 常见问题排查

### 1. 服务返回 500，日志显示 QWEN_API_KEY 为空

说明 API 服务进程没有加载 `.env`，可能回退到了不可用的 Ollama。

处理：

- 确认 `.env` 有 `QWEN_API_KEY`。
- 确认启动 API 时环境变量已加载。
- 用健康检查确认 LLM 是 healthy。

### 2. Qwen 报 Too many requests 或 503

这是外部服务容量或限流问题。

处理：

```bash
--request-retries 5 --retry-backoff 30
```

如果仍频繁失败，等一段时间再跑，或先用 `--skip-ragas` 只检查业务链路。

### 3. Ragas 阶段出现 TimeoutError

少量内部评分任务 timeout 时，Ragas 仍可能完成并输出聚合分数。

处理：

- 先看报告是否完整生成。
- 如果关键指标为 `None` 或大量 `nan`，再重跑。
- 如果只是 1-2 个内部任务 timeout，可以结合人工复核使用结果。

### 4. contexts 为空

Ragas 无法评分。

处理：

- 确认请求传了 `eval_debug=true`。
- 确认 Feishu debug response 带 `contexts`。
- 确认 `/ask` 请求里 `include_contexts=true`。
- 检查该问题是否走了不检索的 route，例如 reset。

### 5. source rate 低

处理：

- 检查 response 的 `sources` 字段是否为空。
- 检查推荐格式是否正确提取 arXiv URL。
- 检查 paper-focused ask 是否把 `/ask` 的 sources 透传回 Feishu response。

### 6. memory pass rate 低

处理：

- 看同一组样本的 `session_id` 是否一致。
- 看脚本是否自动加了 session prefix。
- 看 `intent` 是否识别为追问。
- 看 `rewritten_query` 是否把“第一篇/第二篇/这篇”改写到正确论文。
- 看 reset 后 memory 是否真的清空。

## 推荐的质量判断方式

不要只看一个总分。建议按以下层次判断：

1. 先看是否有接口错误、超时、contexts 为空等基础问题。
2. 再看 rule pass rate、source rate、memory pass rate 是否过线。
3. 再看三个 Ragas 总分是否过线。
4. 最后看失败样本是否集中在某一类任务。

如果总分略有提升，但新增了 reset 或来源失败，通常不能认为质量变好。

如果 Ragas 略有下降，但人工看失败样本发现回答更符合产品预期，可以保留结果并记录原因。

这套评估的核心价值不是给出一个“绝对分数”，而是让每次改动都有可复现的证据链：同一批问题、同一套指标、同一份报告格式、可回看 answer/context/source。

