# Tabular LLM：构建面向表格智能任务的大型语言模型

<img src="./tabular_llm_figures/wechat_group.jpg" width = "200" height = "300" align=right />

本项目基于[**Alpaca-CoT项目**](https://github.com/PhoebusSi/Alpaca-CoT)（一个多接口统一的轻量级LLM指令微调平台），目标是广泛收集开源的表格智能任务数据集（比如表格问答、表格-文本生成等），然后将【原始任务数据】整理为【指令微调格式的数据】并基于Alpaca-CoT项目微调相应的LLM，进而增强LLM对于表格数据的理解，最终构建出专门面向表格智能任务的大型语言模型。

我们目前正在整理学界现有的表格智能数据集，也非常欢迎您向我们提供任何尚未收集的表格相关任务的数据集，我们将努力统一它们的格式并开源训练好的模型。**我们希望本项目能够助力开源社区复现并进一步增强ChatGPT的表格处理能力，同时也使研究者构建针对特定垂类领域的表格智能LLM时，有一个更好的数据和模型基础。**

如果您对“表格+LLM”感兴趣，欢迎您加入本项目的微信群，和更多志同道合的研究者进行讨论交流。


## News

- 2023.05.05：项目公开。

## 0. 背景

以ChatGPT为代表的LLMs对NLP研究领域产生了巨大的冲击，表格智能研究方向也是如此。我们对ChatGPT处理表格的能力进行了初步调研，发现它至少支持以下表格智能任务：

（1）**表格绘制**：根据用户的自然语言描述，精确地生成Markdown格式的表格，如下图所示。

<img src="./tabular_llm_figures/draw_table.png" width = "450" height = "300" align=center />

（2）**表格修改**：根据用户的要求对表格进行修改，比如增加和删除行列等，如下图所示。

<img src="./tabular_llm_figures/table_modification_example.png" width = "400" height = "400" align=center />

(3) **表格问答**：回答用户提出的关于表格的问题，比如“福建力佳公司生产的拖拉机的证书编号和型号分别是什么？”，这是一个陷阱问题，但ChatGPT依然给出了正确的回答，如下图所示。

<img src="./tabular_llm_figures/TQA_example.png" width = "400" height = "130" align=center />

(4) **文本-表格制作**：基于一段文本，制作表格展示文本中的信息，如下图所示。

<img src="./tabular_llm_figures/text-to-table_example.png" width = "700" height = "300" align=center />

(5) **表格事实验证**：用户给出一个陈述语句，验证表格内容对于该语句是支持、否定还是不相关，如下图所示。

<img src="./tabular_llm_figures/TFV_example.png" width = "400" height = "200" align=center />

(6) **表格-文本生成**：生成一段分析和概括表格内容的文本，如下图所示。

<img src="./tabular_llm_figures/table-to-text_example.png" width = "400" height = "150" align=center />

## 1. 动机

虽然ChatGPT已经具备了不错的表格处理能力，但它也存在一些局限：

- **只支持用Markdown格式表示的简单表格**：ChatGPT、文心一言等大模型目前仅支持Markdown格式的表格，第一行是列表头，其余行是数据，不支持包含合并单元格的层级表格。
- **在各项表格智能任务中的能力有待进一步增强**：ChatGPT已经具备多项基本的表格处理能力，但其能力有待进一步增强，比如针对表格问答任务，ChatGPT能够较好地回答关于表格的信息查找类问题（选择表格中的文本作为答案），但不能很好地回答关于表格的数值推理类问题（基于表格中的数值进行多步数学计算）。
- **与表格处理相关的训练数据并未开源**：ChatGPT的训练数据并未开源。为了复现ChatGPT，开源社区目前也已经贡献了许多纯文本任务的宝贵训练数据，但表格智能任务的训练数据相对较少，缺乏统一的整理。

考虑到上述局限，我们提出Tabular-LLM项目，项目的核心计划如下：

1. **探索不同类型表格的表示方法**：训练LLM势必需要将表格转化为一个文本序列，ChatGPT等LLM使用Markdown格式来表示简单表格，但这种方法无法很好地表示更复杂的表格，比如包含合并单元格的层级表格，因此我们需要探索如何（统一）表示不同类型的表格，更多讨论见下一节。
2. **收集并整理涵盖多种类型表格、多种表格智能任务的数据**：考虑学界目前研究较多的表格智能任务，收集开源的数据集并将其转化为指令微调格式的数据，以便用户按需选择。
3. **开源表格智能LLM并进行测试分析**：利用收集到的数据去微调Alpaca-CoT等模型，构建首批面向表格智能任务的开源LLM，在此基础上对训练好的模型进行测试分析，比如测试训练后的模型在学界测试数据集上的表现，后续将相关实验结果整理为文档，希望能为大家提供一些有用的经验。

## 2. 表格的表示方法

为了让LLM理解表格数据，我们需要将半结构化的表格转化为文本序列，这样才能送入模型进行学习，但一个关键问题是，**我们应该采用什么方法来表示表格，才能更有利于模型的理解？**

### 2.1 不同类型的表格

在现实应用中存在不同类型的表格，它们的结构各不相同，按照表头的分布位置，我们可以初步将表格分为以下4种类型，如下图所示：

- **垂直表格**：第一行是列表头，其余行是沿垂直方向排布的数据，这是最基本的表格类型。
- **水平表格**：第一列是行表头，其余列是沿水平方向排布的数据，比如维基百科经常使用水平表格记录人物信息。
- **层级表格**：表头呈现出层级结构，表格中包含合并单元格，比如统计报告和学术论文经常使用层级表格来展示结果，有些层级表格的行表头和列表头可能都存在层级结构，需要同时考虑行列两个方向的表头来理解数据。
- **复杂表格**：上述3种表格的表头只分布于表格的左方或者上方，而复杂表格的表头可以分布于表格的任意位置，尤其是表头的右下方区域，并可能与普通数据混合在一起，比如专业设备的文档中可能利用这类表格记录设备的基本信息、政府部门的登记表格、公司的面试申请表等也多为复杂表格。

![](./tabular_llm_figures/table_types.png)


### 2.2 启发式文本序列格式

和常见的预训练模型一样，学界现有的表格预训练模型同样可以分为判别式模型和生成式模型。

判别式表格预训练模型采用类似BERT的结构（Encoder），典型模型包括[TAPAS](https://arxiv.org/abs/2004.02349)、[TableFormer](https://arxiv.org/abs/2203.00274)、[TABERT](https://arxiv.org/abs/2005.08314)等，它们的目标是学习到好的表格表示来支持下游任务，包括单元格表示向量、列表示向量等。这类模型通常会在BERT原有嵌入层的基础上，通过引入额外的嵌入层来表示表格的结构，包括列ID嵌入、行ID嵌入等，比如TAPAS模型的嵌入层如下所示。
![](./tabular_llm_figures/tapas_table_representation.png)

生成式表格预训练模型采用类似BART、T5的结构（Encoder-Decoder），典型模型为[TAPEX](https://arxiv.org/abs/2107.07653)，当然采用类似GPT的Decoder结构也是可以的，只是没人做hh，它们的目标是采用Seq2Seq的方式直接完成下游任务，比如直接回答关于表格的问题。这类模型在输入端需要采用某种启发式方法，将表格转化为一个文本序列，如下图所示，TAPEX模型需要将表格拉直为：[HEAD] 列表头1 | 列表头2 | … [ROW] 1 第一行第一列的单元格 | 第一行第二列的单元格 | … [ROW] 2 第二行第一列的单元格 | 第二行第二列的单元格 | …  。
![](./tabular_llm_figures/tapex_table_representation.png)

### 2.3 Markdown格式

ChatGPT、文心一言等模型目前应该是采用Markdown格式来表示表格，使用 “|” 来分隔不同的单元格，使用 “-” 来分隔表头所在的第一行和其他行，如下所示。
![](./tabular_llm_figures/markdown_table.png)
可以发现，基于纯文本格式和Markdown格式的表示方法更适合表示结构较为简单的垂直表格或水平表格，无法很好地表示更复杂的表格结构，比如可能包含合并单元格的层级表格和复杂表格。

### 2.4 HTML格式

为了表示更复杂的表格结构，尤其是合并单元格，我们可以使用HTML格式来表示表格，每对【<tr>.....</tr>】标签之间为表格的一行，每对 【<td>....</td>】 标签之间为一行中不同列的单元格，利用【rowspan=m, colspan=n】参数指定某个单元格可以占据m行n列，如下所示。除了能表示合并单元格，HTML还可以设定单元格对齐、单元格背景颜色等表格样式。
![](./tabular_llm_figures/HTML_table.png)

### 2.5 Latex格式

大家在写论文时也会用Latex代码来表示表格，同样可以支持合并单元格，文本对齐等格式，如下所示：
![](./tabular_llm_figures/latex_table.png)

### 2.6 小结

需要注意的是，选择哪种表格表示方法可能还需要考虑底座LLM的代码能力和具体的应用场景。

- **底座LLM的代码能力**：如果你的底座LLM模型具有较强的代码能力，那么选择HTML格式或者Latex格式表示表格可能更为合适，强大如GPT-4，直接给它提供HTML格式的表格，然后进行表格问答都是可以的。但如果你的底座LLM模型的代码理解能力较弱，那么可能更适合使用Markdown格式或者直接用某种启发式格式将表格拉直，目的是构建出更贴近自然语言的表格表示，降低模型学习的难度。
- **具体的应用场景**：“LLM+具体行业应用场景”的结合已经屡见不鲜，由于不少行业会产生大量的表格数据，这也就带来了新的可能性：为某个具体行业甚至是具体软件“量身定制”一个表格智能LLM，从而给用户提供更友好便捷的表格处理方式。在这种情况下，我们或许只能让LLM“迁就”该行业或者该软件使用的特定表格表示方法，然后通过训练数据让LLM理解这种表示方法。
    
    以微软的Excel为例，其背后也会有一套表格表示方法以及操作表格的编程语言VBA，那么为了开发配合Excel使用的LLM，可能就需要收集相应格式的数据来训练LLM理解这种格式的表格和用户需求，然后让LLM直接生成回复或者生成反映用户需求的VBA代码，最后执行代码返回结果。比如，用户可能会输入用自然语言表示的需求“帮我把行表头对应单元格的字体加粗”或者“帮我在表格后面新增一列，计算B列和C列的差值”，表格智能LLM就需要理解Excel表格并生成VBA代码，最终执行代码返回更新后的表格。整体流程可能如下所示：
![](./tabular_llm_figures/potential_tabular_llm_for_excel.png)

回到本项目，由于我们的主要目标是进行表格智能LLM的初步探索，我们更偏向于增强开源LLM的表格处理能力，尚未考虑落地到具体的应用场景，所以**我们仿照ChatGPT，选择利用Markdown格式来表示垂直表格和水平表格，利用HTML格式来表示包含合并单元格的表格，如果原数据集较难转为HTML格式，那么我们将合并单元格拆分为多个相同的子单元格，然后使用Markdown格式进行表示。**（注：后续可能根据实验结果更换表格表示方法，比如统一使用HTML格式表示，我们会尽可能提供更多格式的数据供大家选择。）

## 3. 样本格式

和Alpaca-CoT项目一样，我们将原数据集中的样本整理为统一的格式，如下所示：

```python
[
{
    'instruction': 任务指令 # 不同的表格智能任务对应的指令可能不同
    'input': 输入字符串, # 基于问题、表格、表格标题等信息构造的格式化输入，
			# 以表格问答为例，输入的构造方式为:
                        # f"Table:\n{markdown_table}\nTable title:\n{table_title}\nQuestion:\n{question_text}"
    'output': 输出字符串, # 模型输出
    'table_type': 表格类型, # vertical:垂直表格; horizontal:水平表格; hierarchical:层级表格; complex:复杂表格 
    'task_type': 任务类型, # 比如TQA:表格问答, TFV:表格事实验证
    'dataset': 原始数据集名称 # 比如WikiSQL
}
]
```

以WikiSQL数据集的一个样本为例，instruction为：

```python
"""Please read the following table in Markdown format and then 
answer the question according to the table. Table cells in 
one row are seperated by '|', and different rows are seperated by '\n'."""
```

input为（print后的结果，为了节约空间对表格进行了删减）：

```python
Table:
| Date | Visitor | Score | Home | Decision | Attendance | Record |
| --- | --- | --- | --- | --- | --- | --- |
| March 1 | Los Angeles | 2 – 5 | Colorado | Cloutier | 18007.0 | 26–36–4 |
| March 6 | Ottawa | 0 – 2 | Los Angeles | Ersberg | 17580.0 | 27–37–5 |
| March 8 | Montreal | 5 – 2 | Los Angeles | Ersberg | 18118.0 | 27–38–5 |
| March 10 | Vancouver | 2 – 1 | Los Angeles | Cloutier | 14653.0 | 27–38–6 |
| March 13 | Los Angeles | 4 – 1 | Nashville | Ersberg | 15853.0 | 28–38–6 |
| March 15 | Los Angeles | 0 – 2 | Minnesota | Ersberg | 18568.0 | 28–39–6 |
Question:
On the Date of March 13, who was the Home team?
```

output为：

```python
"Nashvill"
```

## 4. 数据集统计（不断更新ing）

“样本数量”代表本项目基于原始数据集整理格式后获取的样本数量，未填代表待收集。大家可以基于“下载”中的链接获取整理好的json格式的数据。我们遵照原始数据划分分开训练数据和测试数据，以备未来使用测试集测试模型效果（如果有验证集则默认合并至训练数据）。

### 4.1 下载

汇总后的数据下载：

模型下载：

### 4.2 表格问答

| 数据集 | 会议 | 样本数量 | 简介 | 语言 | 论文 | 下载 | 备注 |
| --- | --- | --- | ------------ | :---: | --- | --- |---|
| [WTQ](https://ppasupat.github.io/WikiTableQuestions/) | ACL 2015 | 训练集：17689，测试集：4344 | 从Wikipedia里随机选择超过8行5列的表格，由众包人员提出问题并给出答案。 | 英文 | Compositional Semantic Parsing on Semi-Structured Tables |  |  |
| [AIT-QA](https://github.com/IBM/AITQA) | NAACL 2022 | 训练集：511，测试集：无 | 来自航空公司年报的层级表格 | 英文 | AIT-QA: Question Answering Dataset over Complex Tables in the Airline Industry |  | 将层级表格中的合并单元格拆分为多个子单元格，然后用markdown格式表示。 |
| [TabMCQ](https://allenai.org/data/tablestore-questions) | 2016 | 训练集：1411，测试集：无 | 数据来自于四年级的科学考试，基于表格提出多选问题。 | 英文 | TabMCQ: A Dataset of General Knowledge Tables and Multiple-choice Questions |  |  |
| [FeTaQA](https://github.com/Yale-LILY/FeTaQA) | 2021 | 训练集：8327，测试集：2003 | 数据来自Wikipedia，以往数据集中的答案都比较简单，比如一个单词，本文构造的数据集中，答案是任意长度的句子。 | 英文 | FeTaQA: Free-form Table Question Answering |  |  |
| [TAT-QA](https://nextplusplus.github.io/TAT-QA/) | ACL 2021 | 训练集：14883，测试集：1669 | 需要同时考虑表格和文本信息进行多跳推理，很多样本需要进行数值计算以得到最终答案。数据来自于公司的经济年报。 | 英文 | TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance |  | 对于需要数值计算的样本，在答案中拼接上了具体的计算公式，比如 20 - 5 = 15。 |
| [WikiSQL](https://github.com/salesforce/WikiSQL) | 2018 | 训练集：59736，测试集：14603 | 表格来自Wikipedia | 英文（有一些样本包含中文） | Seq2SQL: Generating Structured Queries from Natural Language using Reinforcement Learning |  | 基于Tapas论文的方法提取出答案文本，后续考虑真正执行SQL语句提取出更精确的答案文本。 |
| [NL2SQL](https://github.com/ZhuiyiTechnology/TableQA) | 2020 | 训练集：45918，测试集：4055 | 来自追一科技，首届中文NL2SQL挑战赛数据集，强调问题中的用词和表格中的用词不一定严格相同，比如“腾讯/鹅厂”，同时也包含无法回答的问题。 | 中文 | TableQA: a Large-Scale Chinese Text-to-SQL Dataset for Table-Aware SQL Generation |  | 执行SQL语句提取答案文本，对于无法回答的问题，答案设置为“根据表格信息无法回答该问题。” |
| [HiTab](https://github.com/microsoft/HiTab) | ACL 2020 |  | 层级表格数据集，包含TQA和Table-to-text两种任务 | 英文 | HiTab : A Hierarchical Table Dataset for Question Answering and Natural Language Generation |  |  |
| [PACIFIC](https://github.com/dengyang17/pacific) | EMNLP 2022 |  | 基于TAT-QA构建的对话数据集 | 英文 | PACIFIC: Towards Proactive Conversational Question Answering over Tabular and Textual Data in Finance |  |  |
| [FINQA](https://github.com/czyssrs/FinQA#finqa) | EMNLP 2021 |  | 面向金融数据的table-text数值推理数据集 | 英文 | FINQA: A Dataset of Numerical Reasoning over Financial Data |  |  |
| [TabMWP](https://promptpg.github.io/index.html#home) | ICLR 2023 |  | 基于表格数据的数学应用题，标注了具体的思维链推导过程。 | 英文 | Dynamic Prompt Learning via Policy Gradient for Semi-structured Mathematical Reasoning |  |  |

### 4.3 表格事实验证

| 数据集 | 会议 | 样本数量 | 简介 | 语言 | 论文 | 下载 | 备注 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [TABFACT](https://tabfact.github.io/) | ICLR 2020 | 训练集：105436，测试集：12839 | 表格来自Wikipedia | 英文 | TabFact: A Large-scale Dataset for Table-based Fact Verification |  |  |
| [Infotab](https://infotabs.github.io/) | ACL 2020 | 训练集：18338，测试集：1800 | 来源于Wikipedia infobox ，属于entity table；三分类；除了train/dev/test还有对抗和跨领域测试集 | 英文 | INFOTABS: Inference on Tables as Semi-structured Data |  |  |
| [PubHealthTab](https://github.com/mubasharaak/PubHealthTab) | NAACL 2022，Findings |  | 三分类；可能有层级表格，表格可能存在列表头和行表头；表格同时给出了html格式和列表格式 | 英文 | PubHealthTab: A Public Health Table-based Dataset for Evidence-based Fact Checking |  |  |

### 4.4 表格→文本生成

### 4.5 文本→表格绘制

### 4.6 表格修改

## 5. 实验分析和经验总结

## 6. 未来计划

- 持续收集更多的表格智能任务数据集
- 对训练好的模型进行测试分析，总结经验供大家参考
- 构建一个在线demo

本文档撰写过程中难免有所纰漏，欢迎大家随时提issue或者在微信群中指出项目中的错误，我们将及时进行订正，感谢大家的阅读！！！
