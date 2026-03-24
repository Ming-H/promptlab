# PromptLab

AI 提示词进化框架 - 通过自动变异、评估和迭代，发现最优的提示词表达方式。

## 特性

- **自动进化**: 使用遗传算法自动探索提示词空间
- **多种变异策略**: 重写、扩展、简化、结构调整等
- **可定制评估**: 使用自己的评估函数量化提示词质量
- **迭代优化**: 保留高分变种，持续进化直到收敛

## 安装

```bash
pip install promptlab
```

## 快速开始

```python
from promptlab import evolve_prompt

# 定义评估函数
def my_evaluator(prompt: str) -> float:
    """
    测试提示词并返回 0-1 之间的分数。
    这里可以用 LLM 评估或实际运行任务来打分。
    """
    # 简单示例：检查关键词覆盖
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

## 高级用法

### 自定义配置

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
    initial_prompt="Write a blog post about AI",
    evaluator=detailed_evaluator,
)

print(f"最佳得分: {result.best_prompt.score}")
print(f"最佳提示词: {result.best_prompt.content}")
```

### 使用不同的 LLM

```python
from promptlab.utils.llm import create_llm_client
from promptlab import EvolutionEngine, EvolutionConfig

# 使用 OpenAI
openai_client = create_llm_client("openai", model="gpt-4o-mini")

# 使用 Anthropic
anthropic_client = create_llm_client("anthropic", model="claude-3-haiku-20240307")

config = EvolutionConfig()
engine = EvolutionEngine(config, llm_client=openai_client)
```

## 变异策略

PromptLab 支持多种提示词变异策略：

| 策略 | 描述 |
|------|------|
| `REWRITE` | 完全重写，保持语义不变 |
| `EXPAND` | 添加细节和上下文 |
| `SIMPLIFY` | 精简表达，去除冗余 |
| `STRUCTURE` | 改变结构（列表、分段、Markdown） |
| `STYLE` | 转换风格（正式、口语、技术性） |
| `EXAMPLE` | 添加或修改示例 |

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 类型检查
mypy promptlab
```

## License

MIT
