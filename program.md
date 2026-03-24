# PromptLab - AI 提示词进化框架

## 项目目标

PromptLab 是一个基于进化算法的 AI 提示词优化框架。通过自动变异、评估和迭代，帮助用户发现最优的提示词表达方式。

### 核心理念

1. **自动化进化** - 让 AI 自动探索提示词空间，无需手动调优
2. **策略多样性** - 支持多种变异策略（重写、扩展、简化、风格转换等）
3. **可评估性** - 通过自定义评估函数量化提示词质量
4. **迭代优化** - 保留高分变种，持续进化直到收敛

## 架构设计

### 模块结构

```
promptlab/
├── core/
│   ├── mutator.py      # 提示词变异引擎
│   ├── evaluator.py    # 评估框架
│   ├── evolution.py    # 进化算法协调器
│   └── types.py        # 类型定义
└── utils/
    └── llm.py          # LLM 调用封装
```

### 模块职责

#### `types.py` - 类型定义
- 定义 `Prompt` 数据类（内容、得分、代数、元数据）
- 定义 `MutationStrategy` 枚举
- 定义 `EvolutionConfig` 配置类
- 定义 `EvaluationResult` 结果类

#### `mutator.py` - 变异引擎
- `Mutator` 类：执行提示词变异
- 支持的策略：
  - `REWRITE` - 完全重写，保持语义
  - `EXPAND` - 添加细节和上下文
  - `SIMPLIFY` - 精简表达，去除冗余
  - `STRUCTURE` - 改变结构（列表、分段、Markdown）
  - `STYLE` - 转换风格（正式、口语、技术性）
  - `EXAMPLE` - 添加/修改示例
- 每次变异调用 LLM 生成新版本

#### `evaluator.py` - 评估框架
- `Evaluator` 类：执行提示词评估
- 接受用户提供的评估函数
- 运行测试用例并收集结果
- 计算得分和指标

#### `evolution.py` - 进化引擎
- `EvolutionEngine` 类：协调整个进化过程
- 管理种群（多个提示词变种）
- 执行选择、变异、评估循环
- 跟踪进化历史和最优解

#### `utils/llm.py` - LLM 封装
- 统一的 LLM 调用接口
- 支持多种模型（OpenAI、Anthropic、本地模型）
- 重试和错误处理
- Token 使用跟踪

### 数据流

```
初始提示词
    ↓
变异（生成 N 个变种）
    ↓
评估（对每个变种打分）
    ↓
选择（保留 top-k）
    ↓
重复（达到代数或收敛）
    ↓
返回最优提示词
```

## API 设计目标

### 简单易用

```python
from promptlab import evolve_prompt

def my_evaluator(prompt: str) -> float:
    # 测试提示词，返回 0-1 的分数
    ...

best_prompt = evolve_prompt(
    initial_prompt="Write a blog post about AI",
    evaluator=my_evaluator,
    generations=10,
    population_size=8
)
```

### 高级用法

```python
from promptlab import EvolutionEngine, Mutator, Evaluator
from promptlab.core.types import EvolutionConfig

config = EvolutionConfig(
    generations=20,
    population_size=10,
    mutation_strategies=["EXPAND", "STRUCTURE", "EXAMPLE"],
    keep_top=3
)

engine = EvolutionEngine(config)
result = engine.evolve(initial_prompt, evaluator)
```

## 评估标准

### 代码质量

1. **功能正确** - 所有测试通过，核心功能完整实现
2. **类型安全** - 完整的类型注解，通过 mypy 检查
3. **代码清晰** - 良好的命名、合理的模块划分、必要的注释
4. **错误处理** - 优雅处理 LLM API 错误、超时等异常

### API 设计

1. **一致性** - 遵循 Python 惯例，直观的方法命名
2. **可扩展** - 易于添加新的变异策略
3. **可配置** - 提供合理的默认值，同时允许深度定制
4. **文档完整** - 详细的 docstring 和 README

### 测试覆盖

1. 单元测试覆盖核心模块
2. 集成测试验证端到端流程
3. Mock LLM 调用以提高测试速度和可靠性

## 迭代策略

### 第一阶段：基础功能
- 实现核心数据类型
- 实现基本的变异和评估
- 验证端到端流程

### 第二阶段：增强功能
- 添加更多变异策略
- 实现种群管理
- 添加进化历史跟踪

### 第三阶段：优化和打磨
- 性能优化（并行评估）
- 更好的错误处理
- 完善文档和示例

## 技术约束

- Python 3.10+
- 最小化外部依赖（仅 LLM SDK）
- 使用标准库和 pydantic 进行数据验证
