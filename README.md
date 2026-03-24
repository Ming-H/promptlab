# PromptLab

[English](#english) | [中文](#中文)

---

<a name="english"></a>
## English

### AI Prompt Evolution Framework - Discover Optimal Prompt Expressions Automatically

PromptLab is a powerful framework that uses genetic algorithms to automatically optimize prompts through mutation, evaluation, and iteration. It helps you find the most effective way to express your prompts for any given task.

### Features

- **Automatic Evolution**: Genetic algorithm-based prompt space exploration
- **Multiple Mutation Strategies**: Rewrite, Expand, Simplify, Structure, Style, Example
- **Customizable Evaluation**: Use your own evaluation functions to score prompt quality
- **Iterative Optimization**: Keep high-scoring variants and evolve until convergence
- **Multi-LLM Support**: Works with OpenAI, Anthropic Claude, and mock clients for testing

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PromptLab Architecture                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌────────────────┐         ┌────────────────┐                     │
│  │  Initial Prompt│────────▶│    Mutator     │                     │
│  │                │         │ (Mutation Engine)│                    │
│  └────────────────┘         └────────┬───────┘                     │
│                                      │                              │
│                                      ▼                              │
│                           ┌────────────────┐                        │
│                           │   Population   │                        │
│                           │  (Variants)    │                        │
│                           └────────┬───────┘                        │
│                                    │                                │
│                                    ▼                                │
│                           ┌────────────────┐                        │
│                           │   Evaluator    │                        │
│                           │  (Scoring)     │                        │
│                           └────────┬───────┘                        │
│                                    │                                │
│                                    ▼                                │
│                           ┌────────────────┐                        │
│                           │   Selection    │                        │
│                           │ (Keep Best)    │                        │
│                           └────────┬───────┘                        │
│                         ┌──────────┴──────────┐                     │
│                         ▼                     ▼                      │
│                  ┌───────────┐         ┌───────────┐               │
│                  │ Next Gen  │         │Best Result│               │
│                  └───────────┘         └───────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### Installation

```bash
pip install promptlab
```

### Quick Start

```python
from promptlab import evolve_prompt

# Define your evaluation function
def my_evaluator(prompt: str) -> float:
    """
    Test the prompt and return a score between 0-1.
    Use LLM evaluation or actual task performance here.
    """
    keywords = ["clear", "specific", "detailed"]
    return sum(1 for k in keywords if k in prompt.lower()) / len(keywords)

# Evolve your prompt
best_prompt = evolve_prompt(
    initial_prompt="Write about machine learning",
    evaluator=my_evaluator,
    generations=10,
    population_size=8,
)

print(f"Best prompt: {best_prompt}")
```

### Advanced Usage

#### Custom Configuration

```python
from promptlab import EvolutionEngine, Evaluator
from promptlab.core.types import EvolutionConfig, MutationStrategy

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
    temperature=0.7,
)

engine = EvolutionEngine(config)

def detailed_evaluator(prompt: str) -> float:
    # Your evaluation logic
    return 0.8

result = engine.evolve(
    initial_prompt="Write a blog post about AI",
    evaluator=detailed_evaluator,
)

print(f"Best score: {result.best_prompt.score}")
print(f"Best prompt: {result.best_prompt.content}")
print(f"Generations: {result.generations_completed}")
```

### Mutation Strategies

| Strategy | Description |
|----------|-------------|
| `REWRITE` | Complete rewrite while preserving semantics |
| `EXPAND` | Add details and context |
| `SIMPLIFY` | Remove redundancy, make concise |
| `STRUCTURE` | Change structure (lists, sections, Markdown) |
| `STYLE` | Transform style (formal, casual, technical) |
| `EXAMPLE` | Add or modify examples |

### Using Different LLMs

```python
from promptlab.utils.llm import create_llm_client
from promptlab import EvolutionEngine, EvolutionConfig

# OpenAI
openai_client = create_llm_client("openai", model="gpt-4o-mini")

# Anthropic
anthropic_client = create_llm_client("anthropic", model="claude-3-haiku-20240307")

# Use with engine
config = EvolutionConfig()
engine = EvolutionEngine(config, llm_client=openai_client)
```

### API Reference

#### `evolve_prompt()` - Convenience Function

```python
def evolve_prompt(
    initial_prompt: str,
    evaluator: Callable[[str], float],
    generations: int = 10,
    population_size: int = 8,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str
```

#### `EvolutionResult`

```python
@dataclass
class EvolutionResult:
    best_prompt: Prompt              # Best prompt found
    all_prompts: list[Prompt]        # All generated prompts
    history: list[dict]              # Evolution history
    generations_completed: int       # Generations run
    converged: bool                  # Whether converged
```

### Project Structure

```
promptlab/
├── promptlab/
│   ├── __init__.py           # Public API
│   ├── core/
│   │   ├── types.py          # Type definitions
│   │   ├── mutator.py        # Mutation engine
│   │   ├── evaluator.py      # Evaluator
│   │   └── evolution.py      # Evolution engine
│   └── utils/
│       └── llm.py            # LLM clients
├── tests/                    # 181 tests
├── examples/
│   └── evolve_prompt.py      # Usage example
└── pyproject.toml
```

### Development

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Type checking
mypy promptlab

# Run with coverage
pytest --cov=promptlab
```

### Test Coverage

- **181 tests** with **88% coverage**
- All tests passing with full type annotations

### License

MIT

---

<a name="中文"></a>
## 中文

### AI 提示词进化框架 - 自动发现最优提示词表达方式

PromptLab 是一个强大的框架，使用遗传算法通过变异、评估和迭代自动优化提示词。它帮助你找到针对任何任务最有效的提示词表达方式。

### 功能特性

- **自动进化**：基于遗传算法的提示词空间探索
- **多种变异策略**：重写、扩展、简化、结构调整、风格转换、添加示例
- **可定制评估**：使用自定义评估函数量化提示词质量
- **迭代优化**：保留高分变种，持续进化直到收敛
- **多 LLM 支持**：支持 OpenAI、Anthropic Claude，以及测试用模拟客户端

### 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PromptLab 架构                               │
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
│                         ┌──────────┴──────────┐                     │
│                         ▼                     ▼                      │
│                  ┌───────────┐         ┌───────────┐               │
│                  │ 下一代     │         │ 最佳结果   │               │
│                  └───────────┘         └───────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 安装

```bash
pip install promptlab
```

### 快速开始

```python
from promptlab import evolve_prompt

# 定义评估函数
def my_evaluator(prompt: str) -> float:
    """
    测试提示词并返回 0-1 之间的分数。
    这里可以用 LLM 评估或实际运行任务来打分。
    """
    keywords = ["clear", "specific", "detailed"]
    return sum(1 for k in keywords if k in prompt.lower()) / len(keywords)

# 进化提示词
best_prompt = evolve_prompt(
    initial_prompt="Write about machine learning",
    evaluator=my_evaluator,
    generations=10,
    population_size=8,
)

print(f"最佳提示词: {best_prompt}")
```

### 进阶使用

#### 自定义配置

```python
from promptlab import EvolutionEngine, Evaluator
from promptlab.core.types import EvolutionConfig, MutationStrategy

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
    temperature=0.7,
)

engine = EvolutionEngine(config)

def detailed_evaluator(prompt: str) -> float:
    # 你的评估逻辑
    return 0.8

result = engine.evolve(
    initial_prompt="写一篇关于 AI 的博客文章",
    evaluator=detailed_evaluator,
)

print(f"最佳得分: {result.best_prompt.score}")
print(f"最佳提示词: {result.best_prompt.content}")
print(f"进化代数: {result.generations_completed}")
```

### 变异策略

| 策略 | 描述 |
|------|------|
| `REWRITE` | 完全重写，保持语义不变 |
| `EXPAND` | 添加细节和上下文 |
| `SIMPLIFY` | 精简表达，去除冗余 |
| `STRUCTURE` | 改变结构（列表、分段、Markdown） |
| `STYLE` | 转换风格（正式、口语、技术性） |
| `EXAMPLE` | 添加或修改示例 |

### 使用不同的 LLM

```python
from promptlab.utils.llm import create_llm_client
from promptlab import EvolutionEngine, EvolutionConfig

# OpenAI
openai_client = create_llm_client("openai", model="gpt-4o-mini")

# Anthropic
anthropic_client = create_llm_client("anthropic", model="claude-3-haiku-20240307")

# 使用引擎
config = EvolutionConfig()
engine = EvolutionEngine(config, llm_client=openai_client)
```

### API 参考

#### `evolve_prompt()` - 便捷函数

```python
def evolve_prompt(
    initial_prompt: str,
    evaluator: Callable[[str], float],
    generations: int = 10,
    population_size: int = 8,
    model: str = "gpt-4o-mini",
    temperature: float = 0.7,
) -> str
```

#### `EvolutionResult`

```python
@dataclass
class EvolutionResult:
    best_prompt: Prompt              # 最佳提示词
    all_prompts: list[Prompt]        # 所有生成的提示词
    history: list[dict]              # 进化历史
    generations_completed: int       # 完成的代数
    converged: bool                  # 是否收敛
```

### 项目结构

```
promptlab/
├── promptlab/
│   ├── __init__.py           # 公开 API
│   ├── core/
│   │   ├── types.py          # 类型定义
│   │   ├── mutator.py        # 变异引擎
│   │   ├── evaluator.py      # 评估器
│   │   └── evolution.py      # 进化引擎
│   └── utils/
│       └── llm.py            # LLM 客户端
├── tests/                    # 181 个测试
├── examples/
│   └── evolve_prompt.py      # 使用示例
└── pyproject.toml
```

### 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 类型检查
mypy promptlab

# 带覆盖率运行
pytest --cov=promptlab
```

### 测试覆盖

- **181 个测试**，**88% 覆盖率**
- 所有测试通过，完整类型注解

### 许可证

MIT
