··# 非遗醒狮文化跨平台舆情分析：基于BERT情感回归与BERTopic主题建模

## 目录
- [1. 项目概述](#1-项目概述)
- [2. 项目亮点与技术栈](#2-项目亮点与技术栈)
- [3. 项目文件结构](#3-项目文件结构)
- [4. 各模块功能详解](#4-各模块功能详解)
  - [4.1 核心数据与模型](#41-核心数据与模型)
  - [4.2 主分析流程 (Step 0-5)](#42-主分析流程-step-0-5)
  - [4.3 进阶分析与可视化](#43-进阶分析与可视化)
- [5. 环境配置与使用指南](#5-环境配置与使用指南)
  - [5.1 环境准备](#51-环境准备)
  - [5.2 运行流程](#52-运行流程)
- [6. 分析结果展示（示例）](#6-分析结果展示示例)
  - [6.1 跨平台方面情感对比](#61-跨平台方面情感对比)
  - [6.2 主题建模距离图](#62-主题建模距离图)
- [7. 未来展望](#7-未来展望)

---

## 1. 项目概述

本项目是一个针对**非物质文化遗产——醒狮文化**在Bilibili和YouTube两大主流视频平台上的公众舆情分析项目。它通过一个端到端的数据科学流程，深度挖掘了公众对醒狮文化的看法、情感倾向和讨论热点。

**核心工作流:**
1.  **情感分析 (Sentiment Analysis)**: 对爬取的视频评论进行人工标注，并微调中文语言模型 **`IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`**。与传统分类任务不同，本项目将其改造为**回归模型**，实现了对评论情感从-1（极度负面）到1（极度正面）的连续、精细化打分。
2.  **方面情感分析 (Aspect-Based Sentiment Analysis, ABSA)**: 基于精心构建的关键词词典，对每一条评论进行方面归类（如“表演技巧”、“文化价值”等），并结合情感分数进行多维度交叉分析。
3.  **主题建模 (Topic Modeling)**: 利用 **`shibing624/text2vec-base-chinese`** 句向量模型和 **BERTopic** 算法，对评论进行无监督主题聚类，自动发现公众讨论的核心话题。

本项目旨在通过现代NLP技术，量化和解读文化现象背后的公众叙事，为文化传播策略提供数据驱动的洞见。

## 2. 项目亮点与技术栈

- **情感回归**: 采用回归任务微调BERT，实现了情感的**量化打分**，比传统的情感分类（正/中/负）更具分析深度。
- **多维度分析**: 结合了情感分数、方面归类和平台来源，产出了如“不同平台在‘文化传承’方面的情感差异”等深度洞察。
- **先进主题建模**: 使用基于句向量的BERTopic，相比传统LDA模型能更好地捕捉语义，主题划分更具可解释性。
- **模块化代码**: 整个项目被拆分为清晰的步骤 (`step0` 到 `step5`) 和独立的进阶分析脚本，结构清晰，易于维护和复现。
- **高级定性探查**: 最终的分析脚本使用句向量模型过滤内容相似的评论，确保挖掘出的典型正/负面评论具有多样性和代表性。

**主要技术栈:**
- **模型框架**: `PyTorch`, `Transformers (Hugging Face)`, `sentence-transformers`
- **核心算法**: `BERTopic`, `UMAP`
- **数据处理**: `pandas`, `numpy`
- **统计分析**: `scipy`
- **数据可视化**: `matplotlib`, `seaborn`, `adjustText`

## 3. 项目文件结构

```
BERT-BERTOPIC/
├── data/                             # 存放原始数据和中间数据
├── models/
│   └── finetuned_sentiment_model/    # 存放微调后的情感回归模型文件
├── results/                          # 存放所有分析结果和图表
│   ├── advanced_analysis/            # 存放进阶分析脚本的产出
│   └── topic_modeling/               # 存放主题建模的结果和模型
├── hit-stopword.txt                  # 哈工大停用词词典
├── general_evaluation_stopwords.txt  # 自定义补充停用词词典                    
├── visualize-sentiment_analysis/     # (旧文件夹，功能已被adv脚本替代)
│
├── step0_finetune_model.py           # 步骤0: 微调情感回归模型
├── step1_predict_scores.py           # 步骤1: 对全部评论进行情感预测
├── step2a-ABSA.py                    # 步骤2a: 进行基于词典的方面分类
├── step2b-ABSAvisualize.py           # 步骤2b: 可视化方面情感分析结果
├── step3_statistical_tests.py        # 步骤3: 平台间情感差异的统计检验
├── step4_BERTopic.py                 # 步骤4: 训练并保存BERTopic模型
├── step5-visualize_top_topics.py     # 步骤5: 可视化主题建模结果
│
├── process_data_analysis.py          #数据预处理及清洗：专为情感分析
├── process_data_topic.py             #数据预处理与清洗：专为主题建模
├── run_adv_1_aspect_distribution.py  # 进阶分析1: 方面分布对比
├── run_adv_2_sentiment_profile.py    # 进阶分析2: 情感剖面密度图
├── run_adv_3_sub_topic_modeling.py   # 进阶分析3: 对特定方面进行二次主题挖掘
├── run_adv_4_probe_extreme_comments.py # 进阶分析4: 探查两极典型评论
│
├── my_custom_dict.txt                # (自定义词典，可能用于分词)
└── requirements.txt                  # 项目依赖库 (需自行生成)
```

## 4. 各模块功能详解

### 4.1 核心数据与模型

- **`data/`**: 存放项目所需的数据。
  - `test.xlsx`: 包含`cleaned_text`和`manual_score`列的**人工标注数据**，是模型微调的输入。
  - `comments_for_sentiment.xlsx`: 清洗后的全部评论数据，用于批量情感预测。
  - `comments_for_bertopic.xlsx`: 为主题建模准备的数据。

- **`models/finetuned_sentiment_model/`**: 存放`step0`脚本训练好的**Erlangshen-Roberta情感回归模型**。这是一个完整的Hugging Face模型，包含配置文件和权重，可直接加载使用。

- **`results/`**: 项目的输出目录。
  - `sentiment_scores.xlsx`: `step1`的输出，包含每条评论及其预测的情感分数。
  - `final_data_with_aspects.xlsx`: `step2a`的输出，在情感分数基础上增加了方面标签，是后续所有分析的核心数据文件。
  - `statistical_report.txt`: `step3`的输出，包含平台间情感差异的统计检验结果。
  - `topic_modeling/`: 存放`step4`训练的BERTopic模型和主题信息表。
  - `advanced_analysis/`: 存放所有`run_adv_`脚本生成的高级图表和报告。

- **`stopword/`**: 存放停用词表，用于在主题建模时过滤无意义的词语，提升主题质量。

### 4.2 主分析流程 (Step 0-5)

这是项目的核心流水线，请按顺序执行。

- **`step0_finetune_model.py`**:
  - **功能**: **模型训练**。读取人工标注的`data/test.xlsx`，加载预训练的`Erlangshen-Roberta`模型，并将其微调为**单输出神经元的回归模型**（使用`MSELoss`损失函数）。
  - **输出**: 将性能最好的模型保存到`models/finetuned_sentiment_model/`。

- **`step1_predict_scores.py`**:
  - **功能**: **批量预测**。加载微调好的模型，对`data/comments_for_sentiment.xlsx`中的所有评论进行情感打分。
  - **输出**: 生成带有`sentiment_score`列的`results/sentiment_scores.xlsx`。

- **`step2a-ABSA.py`**:
  - **功能**: **方面分类**。基于一个内置的、包含多个方面（如“表演-技巧呈现”）及其关键词的词典，通过**正则表达式匹配**，为每条评论打上方面标签。
  - **输出**: 生成带有`aspect`列的`results/final_data_with_aspects.xlsx`。

- **`step2b-ABSAvisualize.py`**:
  - **功能**: **可视化**。读取`step2a`的输出，生成一张出版级质量的**分组箱线图**，直观对比不同平台在不同方面的情感分数分布。

- **`step3_statistical_tests.py`**:
  - **功能**: **统计检验**。使用**Mann-Whitney U检验**（一种非参数检验），判断Bilibili和YouTube在总体上以及在各个方面上的情感分数差异是否具有统计显著性。
  - **输出**: 生成一份详细的文本报告`results/statistical_report.txt`。

- **`step4_BERTopic.py`**:
  - **功能**: **主题模型训练**。使用`text2vec-base-chinese`模型将评论转化为句向量，然后为Bilibili和YouTube**分别训练BERTopic模型**，以挖掘各自平台用户的讨论热点。
  - **输出**: 将训练好的模型对象和主题信息表保存到`results/topic_modeling/`。

- **`step5-visualize_top_topics.py`**:
  - **功能**: **主题可视化**。加载`step4`保存的模型，生成一系列可视化图表，包括**主题频率图、主题词条形图、主题间距离图和层次聚类图**，帮助理解主题结构。

### 4.3 进阶分析与可视化

这些脚本在主流程完成后运行，用于从特定角度进行深度挖掘。

- **`run_adv_1_aspect_distribution.py`**:
  - **功能**: 分析不同平台用户讨论的**方面焦点差异**。计算每个方面在各平台评论中的占比，并生成对比条形图。

- **`run_adv_2_sentiment_profile.py`**:
  - **功能**: 批量为每个方面生成**情感密度图(KDE Plot)**。这比箱线图更能揭示情感分布的精细形态（如单峰、双峰等）。

- **`run_adv_3_sub_topic_modeling.py`**:
  - **功能**: **二次主题挖掘**。针对`step2a`中被归类为“综合讨论”的“长尾”评论，再次运行BERTopic，从中发现更细粒度或新兴的主题。

- **`run_adv_4_probe_extreme_comments.py`**:
  - **功能**: **典型评论探查**。智能地筛选出得分最高和最低的评论。其亮点在于使用了**句向量模型来过滤掉语义上高度相似的评论**，保证了输出结果的多样性，非常适合定性分析。

---

## 5. 环境配置与使用指南

### 5.1 环境准备

1.  **克隆项目**
    ```bash
    git clone [你的项目Git仓库地址]
    cd BERT-BERTOPIC
    ```

2.  **创建并激活虚拟环境** (推荐)
    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate
    ```

3.  **生成并安装依赖**
    *如果你尚未创建`requirements.txt`，可以使用`pipreqs`工具生成:*
    ```bash
    pip install pipreqs
    pipreqs .
    ```
    *然后安装所有依赖:*
    ```bash
    pip install -r requirements.txt
    ```
    *注意：`pytorch`可能需要根据你的CUDA版本单独安装。*

4.  **数据准备**:
    - 确保`data/`目录下有`test.xlsx`（用于训练）和`comments_for_sentiment.xlsx`（用于预测）等必需文件。

### 5.2 运行流程

请严格按照以下顺序执行脚本，以复现完整的分析流程。

**第一部分：主分析流程**
```bash
# 步骤 0: 训练情感模型
python step0_finetune_model.py

# 步骤 1: 批量预测情感分数
python step1_predict_scores.py

# 步骤 2: 进行方面分类并可视化
python step2a-ABSA.py
python step2b-ABSAvisualize.py

# 步骤 3: 执行统计检验
python step3_statistical_tests.py

# 步骤 4 & 5: 主题建模与可视化
python step4_BERTopic.py
python step5-visualize_top_topics.py
```

**第二部分：进阶分析 (在主流程完成后执行)**
```bash
# 运行所有进阶分析脚本
python run_adv_1_aspect_distribution.py
python run_adv_2_sentiment_profile.py
python run_adv_3_sub_topic_modeling.py
python run_adv_4_probe_extreme_comments.py
```

---

## 6. 分析结果展示（示例）


### 6.1 跨平台方面情感对比

![方面情感箱线图](results/aspect_sentiment_boxplot_english.png)

**分析**: 上图展示了Bilibili和YouTube用户在不同讨论方面的情感得分分布。可以观察到，在“阐释-文化价值”方面，两个平台的用户都表现出极高的正面情绪。而在“表演-技巧呈现”方面，Bilibili用户的情感分数中位数略高于YouTube，且分布更为集中，这可能表明...

### 6.2 主题建模距离图

![Bilibili主题距离图](results/visualization_final/bilibili_intertopic_map_publication.png)

**分析**: 这是Bilibili平台评论的主题间距离图。每个圆圈代表一个主题，大小表示其热度，距离远近表示主题间的语义相关性。我们可以清晰地看到几个核心主题簇，例如左上角的“文化传承与民族自豪”主题群，以及右下角的“表演动作与视觉效果”主题群...

---

## 7. 未来展望

- **模型优化**: 尝试更大规模的预训练模型进行微调，或者探索多任务学习，同时进行情感回归和方面分类。
- **方面提取自动化**: 使用模型（如Span-based Extraction）替代当前的关键词匹配方法，以自动发现新的讨论方面。
- **动态演化分析**: 引入时间戳数据，使用动态BERTopic分析公众讨论热点和情感态度随时间的变化趋势。

- **构建交互式仪表盘**: 使用Streamlit或Dash将分析结果封装成一个交互式的Web应用，方便非技术人员探索数据。
