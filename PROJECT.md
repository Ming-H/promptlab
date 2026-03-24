# PromptLab 项目文档

## 项目概述

PromptLab 是一个 AI 提示词进化框架，通过自动变异、评估和迭代，发现最优的提示词表达方式。框架使用遗传算法思想，在提示词空间中进行自动搜索。

### 核心目标

- 自动化提示词优化过程
- 提供多样化的变异策略
- 灵活的评估框架
- 简洁易用的 API

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PromptLab 架构                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐         ┌────────────────┐                     │
│  │  初始提示词     │────────▶│   Mutator      │                     │
│  │ Initial Prompt │         │   变异引擎      │                     │
│  └────────────────┘         └────────┬───────┘                     │
│                                      │                              │
│                                      ▼                              │
│                           ┌────────────────┐                        │
│                           │  变异种群       │                        │
│                           │  Population    │                        │
│                           └────────┬───────┘                        │
│                                    │                                │
│                                    ▼                                │
│                           ┌────────────────┐                        │
│                           │   Evaluator    │                        │
│                           │   评估器        │                        │
│                           └────────┬───────┘                        │
│                                    │                                │
│                                    ▼                                │
│                           ┌────────────────┐                        │
│                           │    Selection   │                        │
│                           │    选择         │                        │
│                           └────────┬───────┘                        │
│                                    │                                │
│                         ┌──────────┴──────────┐                     │
│                         │                     │                      │
│                         ▼                     ▼                      │
│                  ┌───────────┐         ┌───────────┐               │
│                  │ 下一代     │         │ 最佳结果   │               │
│                  │ Next Gen  │         │ Best Result│              │
│                  └───────────┘         └───────────┘               │
│                                                                     │
├─────────────────────────────────────────────────────────────────────┤
│                          组件说明                                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  EvolutionEngine - 进化协调器                                │   │
│  │  • 管理整个进化流程                                          │   │
│  │  • 协调 Mutator 和 Evaluator                                 │   │
│  │  • 记录历史和结果                                            │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Mutator - 提示词变异引擎                                    │   │
│  │  • REWRITE: 完全重写，保持语义不变                            │   │
│  │  • EXPAND: 添加细节和上下文                                  │   │
│  │  • SIMPLIFY: 精简表达，去除冗余                              │   │
│  │  • STRUCTURE: 改变结构（列表、分段、Markdown）               │   │
│  │  • STYLE: 转换风格（正式、口语、技术性）                     │   │
│  │  • EXAMPLE: 添加或修改示例                                   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  Evaluator - 评估框架                                        │   │
│  │  • 接受用户自定义评估函数                                    │   │
│  │  • 返回 0-1 之间的分数                                       │   │
│  │  • 支持详细指标和错误处理                                    │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │  LLMClient - LLM 调用封装                                    │   │
│  │  • MockLLMClient: 测试用模拟客户端                           │   │
│  │  • OpenAIClient: OpenAI API                                  │   │
│  │  • AnthropicClient: Anthropic Claude API                     │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. Mutator (变异引擎)

**文件**: `promptlab/core/mutator.py`

Mutator 负责生成提示词的变异版本，使用不同的策略引导 LLM 创建变体。

```python
from promptlab import Mutator
from promptlab.core.types import MutationStrategy

# 创建 Mutator
mutator = Mutator(
    llm_client=your_llm_client,
    strategies=[
        MutationStrategy.EXPAND,
        MutationStrategy.STRUCTURE,
        MutationStrategy.EXAMPLE,
    ],
    temperature=0.7,
)

# 生成单个变异
mutated = mutator.mutate(original_prompt)

# 生成多个变异
mutations = mutator.mutate_many(original_prompt, n=10)
```

**变异策略**:

| 策略 | 描述 |
|------|------|
| `REWRITE` | 完全重写，保持语义不变 |
| `EXPAND` | 添加细节和上下文 |
| `SIMPLIFY` | 精简表达，去除冗余 |
| `STRUCTURE` | 改变结构（列表、分段、Markdown） |
| `STYLE` | 转换风格（正式、口语、技术性） |
| `EXAMPLE` | 添加或修改示例 |

### 2. Evaluator (评估器)

**文件**: `promptlab/core/evaluator.py`

Evaluator 封装用户自定义的评估逻辑，将提示词转换为分数。

```python
from promptlab import Evaluator

# 简单评分函数
def my_evaluator(prompt: str) -> float:
    """评估提示词质量，返回 0-1 之间的分数"""
    # 你的评估逻辑
    keywords = ["clear", "specific", "detailed"]
    return sum(1 for k in keywords if k in prompt.lower()) / len(keywords)

# 创建 Evaluator
evaluator = Evaluator(my_evaluator)

# 评估单个提示词
result = evaluator.evaluate("Test prompt")
print(result.score)  # 0.0 - 1.0

# 批量评估
results = evaluator.evaluate_many(["Prompt 1", "Prompt 2"])
```

### 3. EvolutionEngine (进化引擎)

**文件**: `promptlab/core/evolution.py`

EvolutionEngine 协调整个进化过程，实现遗传算法的核心逻辑。

```python
from promptlab import EvolutionEngine, EvolutionConfig

# 配置进化参数
config = EvolutionConfig(
    generations=20,           # 进化代数
    population_size=10,       # 每代变异数量
    mutation_strategies=[     # 变异策略
        MutationStrategy.EXPAND,
        MutationStrategy.STRUCTURE,
    ],
    keep_top=3,               # 保留前 N 名
    random_seed=42,           # 随机种子
    model="gpt-4o-mini",      # LLM 模型
    temperature=0.7,          # 变异温度
)

# 创建引擎
engine = EvolutionEngine(config)

# 运行进化
result = engine.evolve(
    initial_prompt="Write about AI",
    evaluator=my_evaluator,
)

# 获取结果
print(f"最佳提示词: {result.best_prompt.content}")
print(f"最佳得分: {result.best_prompt.score}")
print(f"进化代数: {result.generations_completed}")

# 查看历史
for entry in result.history:
    print(f"代 {entry['generation']}: "
          f"最佳={entry['best_score']:.2f}, "
          f"平均={entry['avg_score']:.2f}")
```

### 4. LLM 封装

**文件**: `promptlab/utils/llm.py`

统一的 LLM 调用接口，支持多个提供商。

```python
from promptlab.utils.llm import create_llm_client

# Mock (测试用)
mock_client = create_llm_client("mock")

# OpenAI
openai_client = create_llm_client(
    "openai",
    model="gpt-4o-mini",
    api_key="your-key",
)

# Anthropic
anthropic_client = create_llm_client(
    "anthropic",
    model="claude-3-haiku-20240307",
    api_key="your-key",
)

# 使用客户端
response = openai_client.complete(
    prompt="Hello, world!",
    temperature=0.7,
    max_tokens=1000,
)
```

## API 文档

### 类型定义

**文件**: `promptlab/core/types.py`

#### `MutationStrategy` (枚举)

提示词变异策略。

```python
class MutationStrategy(Enum):
    REWRITE = "rewrite"
    EXPAND = "expand"
    SIMPLIFY = "simplify"
    STRUCTURE = "structure"
    STYLE = "style"
    EXAMPLE = "example"
```

#### `Prompt` (数据类)

表示进化过程中的单个提示词。

```python
@dataclass
class Prompt:
    content: str                    # 提示词内容
    score: float = 0.0              # 评估得分
    generation: int = 0             # 所属代数
    parent_id: str | None = None    # 父代 ID
    metadata: dict[str, Any]        # 元数据

    @property
    def id(self) -> str:            # 唯一标识符
        ...
```

#### `EvaluationResult` (数据类)

提示词评估结果。

```python
@dataclass
class EvaluationResult:
    prompt: Prompt                  # 被评估的提示词
    score: float                    # 得分 (0-1)
    metrics: dict[str, Any]         # 附加指标
    error: str | None = None        # 错误信息
```

#### `EvolutionConfig` (数据类)

进化过程配置。

```python
@dataclass
class EvolutionConfig:
    generations: int = 10                           # 进化代数
    population_size: int = 8                        # 种群大小
    mutation_strategies: list[MutationStrategy]     # 变异策略
    keep_top: int = 3                               # 保留数量
    random_seed: int | None = None                  # 随机种子
    model: str = "gpt-4o-mini"                      # LLM 模型
    temperature: float = 0.7                        # 温度参数
    max_tokens: int = 2000                          # 最大 token 数
```

#### `EvolutionResult` (数据类)

进化过程结果。

```python
@dataclass
class EvolutionResult:
    best_prompt: Prompt                  # 最佳提示词
    all_prompts: list[Prompt]            # 所有生成的提示词
    history: list[dict[str, Any]]        # 进化历史
    generations_completed: int           # 完成的代数
    converged: bool                      # 是否收敛
```

### 公开 API

#### `evolve_prompt()` (便捷函数)

快速进化提示词的便捷函数。

```python
def evolve_prompt(
    initial_prompt: str,
    evaluator: Callable[[str], float],
    generations: int = 10,
    population_size: int = 8,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str:
    """
    进化提示词并返回最佳结果。

    Args:
        initial_prompt: 初始提示词
        evaluator: 评估函数，接受提示词字符串，返回 0-1 之间的分数
        generations: 进化代数
        population_size: 每代种群大小
        model: LLM 模型
        temperature: 变异温度

    Returns:
        进化后的最佳提示词字符串
    """
```

## 使用示例

### 示例 1: 优化代码生成提示词

```python
from promptlab import evolve_prompt

def code_quality_evaluator(prompt: str) -> float:
    """评估代码生成提示词的质量"""
    prompt_lower = prompt.lower()
    criteria = {
        "有函数说明": any(w in prompt_lower for w in ["function", "def", "method"]),
        "有上下文": any(w in prompt_lower for w in ["context", "given"]),
        "有要求": any(w in prompt_lower for w in ["should", "must", "requirement"]),
        "有示例": "example" in prompt_lower,
        "有输出说明": any(w in prompt_lower for w in ["output", "return", "result"]),
        "够详细": len(prompt.split()) > 20,
    }
    return sum(criteria.values()) / len(criteria)

# 进化提示词
best = evolve_prompt(
    initial_prompt="Write a Python function",
    evaluator=code_quality_evaluator,
    generations=5,
    population_size=6,
)

print(f"最佳提示词: {best}")
```

### 示例 2: 使用详细配置

```python
from promptlab import EvolutionEngine, EvolutionConfig, Evaluator
from promptlab.core.types import MutationStrategy

# 创建配置
config = EvolutionConfig(
    generations=20,
    population_size=10,
    mutation_strategies=[
        MutationStrategy.EXPAND,
        MutationStrategy.STRUCTURE,
        MutationStrategy.EXAMPLE,
    ],
    keep_top=3,
    model="gpt-4o-mini",
    temperature=0.8,
)

# 创建引擎
engine = EvolutionEngine(config)

# 定义评估器
evaluator = Evaluator(lambda p: len(p.split()) / 100)

# 运行进化
result = engine.evolve("Write about AI", evaluator)

# 分析结果
print(f"最佳: {result.best_prompt.content}")
print(f"得分: {result.best_prompt.score:.2f}")

# 查看历史
for h in result.history:
    print(f"代 {h['generation']}: "
          f"最佳={h['best_score']:.2f}, "
          f"平均={h['avg_score']:.2f}")
```

### 示例 3: 使用不同的 LLM

```python
from promptlab import EvolutionEngine, EvolutionConfig
from promptlab.utils.llm import create_llm_client

# 使用 OpenAI
openai_client = create_llm_client("openai", model="gpt-4o-mini")

config = EvolutionConfig(model="gpt-4o-mini")
engine = EvolutionEngine(config, llm_client=openai_client)

# 使用 Anthropic
anthropic_client = create_llm_client(
    "anthropic",
    model="claude-3-haiku-20240307"
)
engine_anthropic = EvolutionEngine(
    EvolutionConfig(),
    llm_client=anthropic_client
)
```

### 示例 4: 自定义变异策略

```python
from promptlab import Mutator
from promptlab.core.types import MutationStrategy

# 只使用特定策略
mutator = Mutator(
    llm_client=your_client,
    strategies=[
        MutationStrategy.EXPAND,
        MutationStrategy.EXAMPLE,
    ],
)

# 生成变异
mutations = mutator.mutate_many("Original prompt", n=5)

# 每个变异都有不同的策略
for m in mutations:
    print(f"{m.metadata['mutation_strategy']}: {m.content}")
```

## 开发状态

### 已完成功能

- ✅ 核心变异引擎 (Mutator)
- ✅ 评估框架 (Evaluator)
- ✅ 进化协调器 (EvolutionEngine)
- ✅ LLM 客户端封装 (Mock, OpenAI, Anthropic)
- ✅ 完整类型定义
- ✅ 单元测试 (19 个测试全部通过)
- ✅ 类型检查 (mypy 通过)
- ✅ 基础文档

### TODO 和改进方向

1. **更多变异策略**
   - 添加 COT (思维链) 策略
   - 添加 FEW_SHOT 策略
   - 添加 ROLE_PLAY 角色扮演策略

2. **增强评估能力**
   - 内置 LLM 评估器
   - 支持批量异步评估
   - 评估结果缓存

3. **并行化支持**
   - 多进程变异生成
   - 并行评估
   - 异步 LLM 调用

4. **可视化工具**
   - 进化过程可视化
   - 提示词树状图
   - 得分趋势图

5. **持久化**
   - 保存/加载进化状态
   - 结果导出 (JSON, CSV)
   - 检查点机制

6. **更多 LLM 支持**
   - Gemini
   - Claude Sonnet
   - 本地模型 (Ollama)

## 测试

### 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_evolution.py::TestMutator

# 带覆盖率
pytest --cov=promptlab
```

### 测试覆盖

- 类型定义测试
- 变异引擎测试
- 评估器测试
- 进化引擎测试
- 集成测试

## 类型检查

```bash
mypy promptlab/
```

## 项目结构

```
promptlab/
├── promptlab/
│   ├── __init__.py           # 公开 API
│   ├── core/
│   │   ├── __init__.py
│   │   ├── types.py          # 类型定义
│   │   ├── mutator.py        # 变异引擎
│   │   ├── evaluator.py      # 评估器
│   │   └── evolution.py      # 进化引擎
│   └── utils/
│       ├── __init__.py
│       └── llm.py            # LLM 客户端
├── tests/
│   └── test_evolution.py     # 测试套件
├── examples/
│   └── evolve_prompt.py      # 使用示例
├── README.md                 # 项目说明
├── PROJECT.md                # 本文档
└── pyproject.toml            # 项目配置
```

## 许可证

MIT License
