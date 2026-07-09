# NoneLinear - ReLE Evaluation: Chinese AI Large Model Capability Benchmark (Continuously Updated)
- ReLE (**R**eally R**e**liable **L**ive **E**valuation for LLM), formerly known as CLiB
- Currently includes 391 large models, covering commercial models such as chatgpt, gpt-5.5, Google gemini-3.1-pro, Claude-5, Wenxin ERNIE-X1.1, ERNIE-5.1, qwen3.7-max, qwen3.7-plus, Baichuan, iFlytek Spark, SenseTime senseChat, etc.,
as well as open-source models like hy3, step3.7-flash, kimi-k2.7, ernie4.5, MiniMax-M3, deepseek-v4, Qwen3.6, llama4, Zhipu GLM-5.2, MiMo-V2, LongCat, gemma4, mistral, and more.
- Supports multi-dimensional capability evaluation, including 7 domains: education, healthcare & mental health, finance, law & civil service, reasoning & mathematical computation, language & instruction following, agent & tool use, and nearly 300 sub-dimensions (such as dentistry, high school Chinese, etc.). See our technical report [ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs](https://www.arxiv.org/abs/2601.17399) and media coverage (Synced): [Global 304 Chinese LLMs Tested: No "All-round King", ReLE Solves Evaluation Dilemma with 70% Cost Reduction](https://www.jiqizhixin.com/articles/2026-02-03)
- Not only provides leaderboards, but also a **defect database of over 2 million LLM issues**! Convenient for community research, analysis, and model improvement.
- Free evaluation service for your private LLMs. Contact us (NoneLinear ReLE benchmark team): [Add WeChat](#联系我们非线智能-ReLE-benchmark团队)


# Table of Contents
- [🔄 Recent Updates](#最近更新)
- [⚓ Popular GitHub LLM Evaluation Projects](#GitHub热门大模型评测项目)
- [📝 Basic Model Information](#大模型基本信息)
- [📊 Leaderboards](#-排行榜)
  - [0. Multimodal Leaderboard](#0多模态排行榜)
  - [1. Comprehensive Capability Leaderboard](#1综合能力排行榜)
    - [1.1 Reasoning Model Leaderboard](#11推理类模型排行榜)
    - [1.2 Commercial LLM Leaderboard (including paid APIs for open-source models)](#12商用大模型排行榜含开源模型的付费API)
    - [1.3 Open-source LLM Leaderboard](#13开源大模型排行榜)
  - [2. Education Leaderboard](#2教育排行榜)
    - [2.1 Primary School Subjects](#21-小学学科) &nbsp;|&nbsp; [2.2 Middle School Subjects](#22-初中学科) &nbsp;|&nbsp; [2.3 High School Entrance Exam TODO](#23-中考TODO)
    - [2.4 High School Subjects](#24-高中学科) &nbsp;|&nbsp; [2.5 College Entrance Exam](#25-高考) &nbsp;|&nbsp; [2.6 Higher Education TODO](#26-高等教育TODO)
    - [2.7 Graduate Entrance Exam TODO](#27-考研TODO) &nbsp;|&nbsp; [2.8 Teacher Qualification TODO](#28-教师资格TODO)
  - [3. Healthcare & Mental Health Leaderboard](#3医疗与心理健康排行榜)    
    - [3.1 Physician](#31-医师) &nbsp;|&nbsp; [3.2 Nursing](#32-护理) &nbsp;|&nbsp; [3.3 Pharmacist](#33-药师)
    - [3.4 Medical Technology](#34-医技) &nbsp;|&nbsp; [3.5 Basic Medical Knowledge](#35-医学基础知识) &nbsp;|&nbsp; [3.6 Medical Graduate Exam](#36-医学考研)
    - [3.7 Mental Health](#37-心理健康)
  - [4. Finance Leaderboard](#4金融排行榜)
    - [4.1 Accounting](#41-财务) &nbsp;|&nbsp; [4.2 Banking](#42-银行) &nbsp;|&nbsp; [4.3 Insurance](#43-保险)
    - [4.4 Securities](#44-证券) &nbsp;|&nbsp; [4.5 Other Financial Qualification Exams](#45-其他金融资格考试) &nbsp;|&nbsp; [4.6 Basic Financial Knowledge](#46-金融基础知识)
    - [4.7 Financial Applications](#47-金融应用)
  - [5. Law & Civil Service Leaderboard](#5法律与行政公务排行榜)
    - [5.1 Bar Exam](#51-律师资格考试)
    - [5.2 Civil Service Exam](#52-公务员考试)
  - [6. Reasoning & Mathematical Computation Leaderboard](#6推理与数学计算排行榜)
    - [6.1 Deductive Reasoning](#61-演绎推理)  &nbsp;|&nbsp; [6.2 Commonsense Reasoning](#62-常识推理) &nbsp;|&nbsp; [6.3 Symbolic Reasoning BBH](#63-符号推理BBH)
    - [6.4 Arithmetic Ability](#64-算术能力) &nbsp;|&nbsp; [6.5 Table QA](#65-表格问答) &nbsp;|&nbsp; [6.6 Table Summarization](#66-表格总结)
    - [6.7 High School Math Olympiad](#67-高中奥数) &nbsp;|&nbsp; [6.8 Middle School Math Olympiad TODO](#68-初中奥数TODO) &nbsp;|&nbsp; [6.9 Primary School Math Olympiad](#69-小学奥数)
    - [6.10 Map Reasoning TODO](#610-地图推理TODO) &nbsp;|&nbsp; [6.11 Spatial Reasoning TODO](#611-空间推理TODO) &nbsp;|&nbsp; [6.12 Sudoku](#612-数独)
    - [6.13 Amount Upper-Lower Case Conversion TODO](#613-金额大小写转换TODO) &nbsp;|&nbsp; [6.14 Date Calculation TODO](#614-日期计算TODO)
  - [7. Language & Instruction Following Leaderboard](#7语言与指令遵从排行榜)
    - [7.1 Idiom Understanding](#71-成语理解) &nbsp;|&nbsp; [7.2 Sentiment Analysis](#72-情感分析) &nbsp;|&nbsp; [7.3 Textual Entailment](#73-文本蕴含) 
    - [7.4 Text Classification](#74-文本分类) &nbsp;|&nbsp; [7.5 Information Extraction](#75-信息抽取) &nbsp;|&nbsp; [7.6 Reading Comprehension](#76-阅读理解) 
    - [7.7 Pronoun Understanding](#77-代词理解) &nbsp;|&nbsp; [7.8 Poetry Matching](#78-诗词匹配) &nbsp;|&nbsp; [7.9 Chinese Instruction Following](#79-中文指令遵从) 
    - [7.10 Chinese Character Forms](#710-汉字字形) &nbsp;|&nbsp; [7.11 Chinese Pinyin TODO](#711-汉语拼音TODO) &nbsp;|&nbsp; [7.12 Find Wrong Characters TODO](#712-找错别字TODO) 
    - [7.13 Sentence Understanding TODO](#713-句子理解TODO) &nbsp;|&nbsp; [7.14 Punctuation TODO](#714-标点符号TODO) &nbsp;|&nbsp; [7.15 Simplified-Traditional Conversion TODO](#715-汉字繁简转换TODO) 
    - [7.16 Language Identification TODO](#716-语种识别TODO)
  - [8. Agent & Tool Use Leaderboard](#8agent与工具调用排行榜)
    - [8.1 TAU](#81-TAU)
    - [8.2 BFCL-V3](#82-BFCL-V3)
  - [9. Coding Leaderboard](#9coding排行榜)
    - [9.1 livecodebench](#91-livecodebench)
    - [9.2 Terminal-Bench-2.0](#92-Terminal-Bench-20)  
  - [10. Integrated LMArena and AA Scores](#10整合LMArena和AA分数)    
- [🌐 Capability Scores](#🌐各项能力评分)
- [Why Make a Leaderboard?](#为什么做榜单)
- [LLM Selection & Evaluation Community](#大模型评测交流群)
- [Cite Us](#如何引用-ReLE-评测Cite-Us)

# Recent Evaluation Updates
- [2026/7/9] v5.10.13
  - New model: hy3
- [2026/7/2] v5.10.12
  - New model: claude-sonnet-5-thinking
- [2026/6/27] v5.10.12
  - New models: doubao-seed-2-1-pro-260628, doubao-seed-2-1-turbo-260628, doubao-seed-evolving
- [2026/6/18] v5.10.11
  - New model: glm-5.2
- [2026/6/16] v5.10.10
  - New model: kimi-k2.7-code
- [2026/6/2] v5.10.9
  - New models: MiniMax-M3, qwen3.7-plus, step-3.7-flash, claude-opus-4.8-thinking
- [2026/5/30] v5.10.8
  - New model: claude-opus-4.8
- [2026/5/23] v5.10.7
  - New model: qwen3.7-max
- [2026/5/21] v5.10.6
  - New model: gemini-3.5-flash
- [2026/5/13] v5.10.5
  - New model: ernie-5.1
- [2026/5/1] v5.10.4
  - New model: qwen3.6-27b
- [2026/4/25] v5.10.3
  - New models: deepseek-v4-flash, deepseek-v4-pro, gpt-5.5
- [2026/4/23] v5.10.2
  - New models: mimo-v2.5, mimo-v2.5-pro
- [2026/4/21] v5.10.1
  - New models: qwen3.6-max-preview, kimi-k2.6
  - Model update: Updated kimi-k2.5 evaluation results (fixed error where reasoning_content was not passed to tool call), scores and rankings have changed
- [2026/4/18] v5.10, [2026/4/15] v5.9, [2026/4/8] v5.8.23, [2026/4/6] v5.8.22, [2026/4/3] v5.8.21, [2026/3/19] v5.8.20, [2026/3/18] v5.8.19, [2026/3/17] v5.8.18, [2026/3/5] v5.8.17, [2026/2/25] v5.8.16, [2026/2/20] v5.8.15, [2026/2/14] v5.8.14, [2026/2/9] v5.8.13, [2026/2/2] v5.8.12, [2026/1/27] v5.8.11, [2026/1/22] v5.8.10, [2025/12/24] v5.8.9, [2025/12/23] v5.8.8, [2025/12/18] v5.8.7, [2025/12/13] v5.8.6, [2025/12/6] v5.8.5, [2025/12/3] v5.8.4, [2025/11/3] v5.8, [2025/10/24] v5.7, [2025/10/13] v5.6, [2025/9/30] v5.5, [2025/9/22] v5.4, [2025/9/14] v5.3, [2025/9/10] v5.2, [2025/9/6] v5.1, [2025/9/1] v5.0, [2025/8/26] v4.13, [2025/8/20] v4.12, [2025/8/15] v4.11, [2025/8/10] v4.10, [2025/8/7] v4.9, [2025/8/1] v4.8, [2025/7/29] v4.7, [2025/7/26] v4.6, [2025/7/23] v4.5, [2025/7/17] v4.4, [2025/7/13] v4.3, [2025/7/12] v4.2, [2025/7/9] v4.1, [2025/7/2] v4.0, [2025/6/23] v3.33, [2025/6/18] v3.32, [2025/6/16] v3.31, [2025/6/13] v3.30, [2025/6/9] v3.29, [2025/6/4] v3.28, [2025/5/29] v3.27, [2025/5/23] v3.26, [2025/5/18] v3.25, [2025/5/15] v3.24, [2025/5/10] v3.23, [2025/5/5] v3.22, [2025/5/2] v3.21, [2025/4/30] v3.20, [2025/4/28] v3.19, [2025/4/22] v3.18, [2025/4/17] v3.17, [2025/4/9] v3.16, [2025/4/5] v3.15, [2025/4/3] v3.14, [2025/3/31] v3.13, [2025/3/29] v3.12, [2025/3/27] v3.11, [2025/3/25] v3.10, [2025/3/23] v3.9, [2025/3/21] v3.8, [2025/3/19] v3.7, [2025/3/17] v3.6, [2025/3/15] v3.5, [2025/3/13] v3.4, [2025/3/11] v3.3, [2025/3/10] v3.2, [2025/3/7] v3.1, [2025/3/4] v3.0, [2025/3/3] v2.22, [2025/2/28] v2.21, [2025/2/24] v2.20, [2025/2/22] v2.19, [2025/2/18] v2.18, [2025/2/14] v2.17, [2025/2/13] v2.16, [2025/2/12] v2.15, [2025/2/10] v2.14, [2025/1/29] v2.13, [2025/1/25] v2.12, [2025/1/23] v2.11, [2025/1/22] v2.10, [2025/1/20] v2.9, [2025/1/17] v2.8, [2025/1/7] v2.7
- 2024: [2024/12/28] v2.6, [2024/12/27] v2.5, [2024/12/25] v2.4, [2024/10/20] v2.3, [2024/9/29] v2.2, [2024/8/27] v2.1, [2024/8/7] v2.0, [2024/7/26] v1.21, [2024/7/15] v1.20, [2024/6/29] v1.19, [2024/6/2] v1.18, [2024/5/8] v1.17, [2024/4/13] v1.16, [2024/3/20] v1.15, [2024/2/28] v1.14, [2024/1/29] v1.13
- 2023: [2023/12/10] v1.12, [2023/11/22] v1.11, [2023/11/5] v1.10, [2023/10/11] v1.9, [2023/9/13] v1.8, [2023/8/29] v1.7, [2023/8/13] v1.6, [2023/7/26] v1.5, [2023/7/18] v1.4, [2023/7/2] v1.3, [2023/6/17] v1.2, [2023/6/10] v1.1, [2023/6/4] v1.0

For details of each version update: [CHANGELOG](CHANGELOG.md)
<br><br>


# Popular GitHub LLM Evaluation Projects
| repo                                                                               | star  | area   | about                                                                                                                                                                                                                                                                   |
|------------------------------------------------------------------------------------|-------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [langfuse](https://github.com/langfuse/langfuse)                                   | 23.6k | Global | Open source LLM engineering platform: LLM Observability, metrics, evals, prompt management, playground, datasets. Integrates with OpenTelemetry, Langchain, OpenAI SDK, LiteLLM, and more. 🍊YC W23                                                                     |
| [opik](https://github.com/comet-ml/opik)                                           | 18.4k | Global | Debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards.                                                                                              |
| [deepeval](https://github.com/confident-ai/deepeval)                      | 14.2k | Global | The LLM Evaluation Framework                                                                                                                                                                                                                                            |
|……|……|……|……|
| [⭐chinese-llm-benchmark（Us）](https://github.com/jeinlee1991/chinese-llm-benchmark) | 5.7k  | **China** | ReLE Chinese LLM Capability Benchmark (Continuously Updated) |                                                                                               |
|……|……|……|……|

See [hot50](GitHub热门评测repo.md)
<br><br>


# Basic Model Information
- [Latest Models This Week](每周最新模型.md)
    - [June 15~June 21](每周最新模型.md#6月156月21)
    - [June 8~June 14](每周最新模型.md#6月86月14)
    - [June 1~June 7](每周最新模型.md#6月16月7)
    - [May 25~May 31](每周最新模型.md#5月255月31)
- For more information, see [Model List](https://nonelinear.com/static/models.html)
<br><br>

# 🚀 Unified LLM Gateway
Introducing the one-stop AI Model Marketplace 🛒, offering the most comprehensive selection of large models so you can always stay ahead.
- 🌐 Global models, all in one place: GPT-5.5, Gemini-3.1-Pro, Claude-4.7, DeepSeek-v4, Kimi-k2.5, and more...
- ⚖️ Intelligent load balancing & high concurrency: We aggregate multiple top providers and achieve automatic load balancing via smart routing. Say goodbye to annoying Rate Limit errors and easily handle any traffic surge!
- 🔀 Automatic failover: Is a single provider's API temporarily "down"? No worries! Our system will seamlessly switch to a healthy backup channel in milliseconds, ensuring your service is 99.9999% highly available and your users never face "service unavailable" embarrassment.
- 🛡️ Online monitoring & intelligent model selection: Seamlessly connect with online performance monitoring tools to close the loop on model selection and evaluation. Let real data speak, helping you easily find the best-performing and most cost-effective model solution.
[How to connect to online performance monitoring](https://nonelinear.com/static/online-eval.html), [How to connect to model selection evaluation](https://nonelinear.com/static/task-create.html)
- 💰 Super cost-effective! ☛[View all models and prices](https://nonelinear.com/static/models.html)
```
from openai import OpenAI
base_url = "https://api.nonelinear.com/v1"
api_key = "<your api key>" # Get it at https://nonelinear.com/static/apikey.html
client = OpenAI(api_key=api_key, base_url=base_url)
client.chat.completions.create(
    model="<model id>", # Model list: https://nonelinear.com/static/models.html
    messages=[{"role": "user", "content": "<your prompt>"}],
)
```
<br><br>


# 💥 Model Selection: Target 90% Cost Reduction
Say NO to "blindly picking" LLMs 🎉! Upload your [custom test data] 📊, and in 5 minutes 🔍 find out which model performs best 🏆 and is most cost-effective 💰 for your scenario! Choose the most suitable model and reduce costs by up to 90% 💥! [Try it now >>](https://nonelinear.com/static/task-create.html)
![link](docs/modelSelection/img/task-result-html.png)
<video controls src="docs/modelSelection/img/modelsel.mp4"></video>

Examples:
- [Table Summarization for WeChat Article Writing](docs/modelSelection/微信文章撰写之表格总结.md)
- [MathML to LaTeX Format](docs/modelSelection/MathML转LaTeX格式.md)
<br><br>


# 📊 Leaderboard
## 0. Multimodal Leaderboard
For detailed data, see [Multimodal Evaluation](README-多模态评测.md)<br>
<br><br>

## 1. Comprehensive Ability Leaderboard
Scoring method for "Comprehensive Ability": "Comprehensive Ability" is now a weighted sum of "Professional Ability" and "General Ability", with weights of 0.3 and 0.7 respectively; "Professional Ability" is the average score of the four domains: "Education", "Healthcare & Mental Health", "Finance", and "Law & Public Administration". "General Ability" is the average score of the four domains: "Reasoning & Mathematical Calculation", "Language & Instruction Following", "Agent & Tool Use", and "Coding".
![link](pic/总分.png)

|Category|Organization|Large Model|[Total Score] Accuracy|Average Time|Average Token Usage|Cost per 1K (CNY)|Ranking (Accuracy)|
|---|---|-----|-------------------|-------|-----------|-----------|-----------|
|Commercial|Alibaba|qwen3.7-max(new)|76.9%|51s|2920|99.0|1|
|Commercial|Doubao|doubao-seed-evolving(new)|75.5%|267s|10392|304.7|2|

For detailed data, see: [Comprehensive Ability Leaderboard](leaderboard/总分.md) | [General Ability Leaderboard](leaderboard/通用能力.md) | [Professional Ability Leaderboard](leaderboard/专业能力.md)
<br><br>

#### 1.1. Reasoning Model Leaderboard
See [Reasoning Model Leaderboard](leaderboard/reasonmodel.md)<br>
<br>
#### 1.2. Commercial Large Model Leaderboard (including paid APIs for open-source models)
[Commercial Large Models with Output Price ≥ 5 CNY](leaderboard/commerce1.md) | [Commercial Large Models with Output Price 1~5 CNY](leaderboard/commerce2.md) | [Commercial Large Models with Output Price < 1 CNY](leaderboard/commerce3.md)<br>
DIY custom dimension leaderboard: ☛ [link](https://nonelinear.com/static/benchmarking.html)
<br>
<br>
#### 1.3. Open-Source Large Model Leaderboard
[Open-Source Large Models under 5B](leaderboard/opensource1.md) | [Open-Source Large Models 5B~20B](leaderboard/opensource2.md) | [Open-Source Large Models over 20B](leaderboard/opensource3.md)<br>
DIY custom dimension leaderboard: ☛[link](https://nonelinear.com/static/benchmarking.html)

<br><br>

## 2. Education Leaderboard
☛☛See the complete leaderboard at [Education](leaderboard/教育.md)<br>

### 2.1 Primary School Subjects
☛☛See the complete leaderboard at [Primary School Subjects](leaderboard/小学学科.md).<br>
Chinese: [Leaderboard](leaderboard/PrimarySchoolChinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolChinese),
English: [Leaderboard](leaderboard/PrimarySchoolEnglish.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolEnglish),
Mathematics: [Leaderboard](leaderboard/PrimarySchoolMathematics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolMathematics),
Morality and Law: [Leaderboard](leaderboard/PrimarySchoolEthics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolEthics),
Science: [Leaderboard](leaderboard/PrimarySchoolScience.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolScience)
<br><br>

### 2.2 Middle School Subjects
☛☛See the complete leaderboard at [Middle School Subjects](leaderboard/初中学科.md).<br>
Biology: [Leaderboard](leaderboard/MiddleSchoolBiology.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolBiology),
Chemistry: [Leaderboard](leaderboard/MiddleSchoolChemistry.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolChemistry),
Chinese: [Leaderboard](leaderboard/MiddleSchoolChinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolChinese),
English: [Leaderboard](leaderboard/MiddleSchoolEnglish.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolEnglish),
Geography: [Leaderboard](leaderboard/MiddleSchoolGeography.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolGeography),
History: [Leaderboard](leaderboard/MiddleSchoolHistory.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolHistory),
Mathematics: [Leaderboard](leaderboard/MiddleSchoolMathematics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolMathematics),
Physics: [Leaderboard](leaderboard/MiddleSchoolPhysics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolPhysics),
Politics: [Leaderboard](leaderboard/MiddleSchoolPolitics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolPolitics)
<br><br>

### 2.3 High School Entrance Exam TODO

### 2.4 High School Subjects
☛☛See the complete leaderboard at [High School Subjects](leaderboard/高中学科.md).<br>
Biology: [Leaderboard](leaderboard/HighSchoolBiology.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolBiology),
Chemistry: [Leaderboard](leaderboard/HighSchoolChemistry.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolChemistry),
Chinese: [Leaderboard](leaderboard/HighSchoolChinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolChinese),
English: [Leaderboard](leaderboard/HighSchoolEnglish.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolEnglish),
Geography: [Leaderboard](leaderboard/HighSchoolGeography.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolGeography),
History: [Leaderboard](leaderboard/HighSchoolHistory.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolHistory),
Mathematics: [Leaderboard](leaderboard/HighSchoolMathematics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolMathematics),
Physics: [Leaderboard](leaderboard/HighSchoolPhysics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolPhysics),
Politics: [Leaderboard](leaderboard/HighSchoolPolitics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolPolitics)
<br><br>

### 2.5 College Entrance Exam (Gaokao)
Past years' Gaokao real exam questions, including simple questions, fill-in-the-blank, multiple choice, etc., only objective questions are retained. All scores are accuracy rates, 100% means all correct; for example, Mathematics 100 means all answers are correct. ☛☛See the complete leaderboard at [Gaokao](leaderboard/高考.md).<br>
(1) 2025 Gaokao<br>
Biology: [Leaderboard](leaderboard/2025高考生物.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考生物),
Chemistry: [Leaderboard](leaderboard/2025高考化学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考化学),
Chinese: [Leaderboard](leaderboard/2025高考语文.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考语文),
English: [Leaderboard](leaderboard/2025高考英语.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考英语),
Geography: [Leaderboard](leaderboard/2025高考地理.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考地理),
History: [Leaderboard](leaderboard/2025高考历史.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考历史),
Mathematics: [Leaderboard](leaderboard/2025高考数学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考数学),
Physics: [Leaderboard](leaderboard/2025高考物理.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考物理),
Politics: [Leaderboard](leaderboard/2025高考政治.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考政治).

(2) 2024 and earlier Gaokao<br>
Biology: [Leaderboard](leaderboard/gaokao-biology.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-biology),
Chemistry: [Leaderboard](leaderboard/gaokao-chemistry.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-chemistry),
Chinese: [Leaderboard](leaderboard/gaokao-chinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-chinese),
Geography: [Leaderboard](leaderboard/gaokao-geography.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-geography),
History: [Leaderboard](leaderboard/gaokao-history.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-history),
Mathematics: [Leaderboard](leaderboard/gaokao-math.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-math),
Physics: [Leaderboard](leaderboard/gaokao-physics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-physics),
Politics: [Leaderboard](leaderboard/gaokao-politics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-politics).
<br><br>

### 2.6 Higher Education TODO
### 2.7 Postgraduate Entrance Exam TODO
### 2.8 Teacher Qualification Exam TODO
<br><br><br>

## 3. Medical and Mental Health Leaderboard
☛☛ For the complete leaderboard, see [Medical and Mental Health](leaderboard/医疗与心理健康.md)<br>

### 3.1 Physicians
☛☛ For the complete leaderboard, see [Physicians](leaderboard/医师.md)<br>
(1) Internal Medicine, [Leaderboard](leaderboard/内科.md)<br>
Completion of standardized residency in internal medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-内科),
Attending physician of traditional Chinese internal medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医内科主治医师),
Attending physician of internal medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内科主治医师),
Attending physician of cardiovascular and respiratory medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心血管内科与呼吸内科主治医师),
Attending physician of nephrology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肾内科主治医师),
Attending physician of gastroenterology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=消化内科主治医师),
Attending physician of integrated Chinese and Western internal medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中西医结合内科主治医师),
Senior title in gastroenterology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=消化内科高级职称),
Senior title in general internal medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=普通内科高级职称),
Senior title in respiratory medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=呼吸内科高级职称),
Senior title in cardiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心内科高级职称),
Attending physician of tuberculosis: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=结核病主治医师),
Senior title in endocrinology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内分泌科高级职称)
<br>

(2) Surgery, [Leaderboard](leaderboard/外科.md)<br>
Completion of standardized residency in surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-外科),
Attending physician of oral and maxillofacial surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔颌面外科主治医师),
Attending physician of plastic surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=整形外科主治医师),
Attending physician of surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=外科主治医师),
Senior title in general surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=普通外科高级职称),
Completion of standardized residency in orthopedics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-骨科),
Intermediate title in orthopedics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=骨科中级职称),
Senior title in orthopedics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=骨科高级职称)
<br>

(3) Obstetrics and Gynecology, [Leaderboard](leaderboard/妇产科.md)<br>
Completion of standardized residency in obstetrics and gynecology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-妇产科),
Attending physician of obstetrics and gynecology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=妇产科主治医师),
Associate chief and chief physician title exam in obstetrics and gynecology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=妇产科学副主任、主任医师职称考试)
<br>

(4) Pediatrics, [Leaderboard](leaderboard/儿科.md)<br>
Completion of standardized residency in pediatrics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-儿科),
Attending physician of pediatrics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=儿科主治医师),
Completion of standardized residency in pediatric surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-小儿外科) 
<br>

(5) Ophthalmology, [Leaderboard](leaderboard/眼科.md)<br>
Completion of standardized residency in ophthalmology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-眼科),
Attending physician of ophthalmology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=眼科主治医师)
<br>

(6) Stomatology, [Leaderboard](leaderboard/口腔科.md)<br>
Completion of standardized residency in stomatology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-口腔科),
Assistant practicing dentist: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔执业助理医师),
Practicing dentist: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔执业医师),
Attending physician of oral medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔内科主治医师),
Attending physician of stomatology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔科主治医师),
Attending physician of prosthodontics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔修复科主治医师),
Attending physician of orthodontics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔正畸学主治医师)
<br>

(7) Otorhinolaryngology, [Leaderboard](leaderboard/耳鼻咽喉科.md)<br>
Completion of standardized residency in otorhinolaryngology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-耳鼻咽喉科),
Attending physician of otorhinolaryngology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=耳鼻咽喉科主治医师)
<br>

(8) Neurology and Psychiatry, [Leaderboard](leaderboard/脑系科.md)<br>
Completion of standardized residency in neurology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-神经内科),
Attending physician of neurology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=神经内科主治医师),
Completion of standardized residency in psychiatry: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-精神科),
Attending physician of psychiatry: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=精神病学主治医师),
Attending physician of psychotherapy: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理治疗学主治医师考试),
Psychological counselor: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理咨询师考试)
<br>

(9) Dermatology, [Leaderboard](leaderboard/皮肤科.md)<br>
Completion of standardized residency in dermatology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-皮肤科),
Intermediate title in dermatology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=皮肤科中级职称),
Attending physician of dermatology and venereology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=皮肤与性病学主治医师)
<br>

(10) Traditional Chinese Medicine and Integrated Chinese and Western Medicine, [Leaderboard](leaderboard/中医与中西医结合.md)<br>
Assistant practicing physician of integrated Chinese and Western medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中西医结合执业助理医师),
Assistant practicing physician of traditional Chinese medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医执业助理医师),
Practicing physician of integrated Chinese and Western medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中西医结合执业医师),
Practicing physician of traditional Chinese medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医执业医师),
Attending physician of acupuncture and moxibustion: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医针灸主治医师)
<br>

(11) Rehabilitation Medicine, [Leaderboard](leaderboard/康复医学科.md)<br>
Completion of standardized residency in rehabilitation medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-康复医学科),
Attending physician of rehabilitation medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学主治医师)
<br>

(12) General Practice, [Leaderboard](leaderboard/全科医学科.md)<br>
Completion of standardized residency in general practice: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-全科医学科),
Attending physician of general practice: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=全科主治医师)
<br>

(13) Clinical Nutrition and Critical Care Medicine, [Leaderboard](leaderboard/临床营养与重症医学.md)<br>
Assistant practicing physician of clinical medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床执业助理医师),
Practicing physician of clinical medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床执业医师),
Attending physician of rheumatology and clinical immunology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=风湿与临床免疫主治医师),
Attending physician of critical care medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=重症医学主治医师),
Attending physician of nutrition: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=营养学主治医师),
Completion of standardized residency in clinical pathology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-临床病理科)
<br>

(14) Oncology, [Leaderboard](leaderboard/肿瘤科.md)<br>
Attending physician of oncology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤学主治医师)
<br>

(15) Anesthesiology and Pain Medicine, [Leaderboard](leaderboard/麻醉疼痛科.md)<br>
Completion of standardized residency in anesthesiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-麻醉科),
Attending physician of anesthesiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=麻醉科主治医师),
Attending physician of pain medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=疼痛科主治医师)
<br>

(16) Public Health and Occupational Diseases, [Leaderboard](leaderboard/公共卫生与职业病.md)<br>
Assistant practicing physician of public health: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公共卫生执业助理医师),
Practicing physician of public health: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公共卫生执业医师),
Intermediate title in hospital infection: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医院感染中级职称),
Attending physician of infectious diseases: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=传染病主治医师),
Attending physician of preventive medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=预防医学主治医师),
Intermediate title in infectious diseases: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=传染病学中级职称),
Attending physician of occupational diseases: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=职业病主治医师)
<br><br>


### 3.2 Nursing
☛☛ For the complete leaderboard, see [Nursing](leaderboard/护理.md)<br>
Nurse qualification exam: [Leaderboard](leaderboard/护士执业资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=护士执业资格考试),
Nurse practitioner qualification exam: [Leaderboard](leaderboard/护师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=护师资格考试),
Pediatric head nurse: [Leaderboard](leaderboard/儿科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=儿科主管护师),
Internal medicine nursing: [Leaderboard](leaderboard/主管护师-内科护理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管护师-内科护理学),
Obstetrics and gynecology nursing: [Leaderboard](leaderboard/主管护师-妇产科护理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管护师-妇产科护理学),
Obstetrics and gynecology head nurse: [Leaderboard](leaderboard/妇产科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=妇产科主管护师),
Surgical head nurse: [Leaderboard](leaderboard/外科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=外科主管护师),
Head nurse qualification exam: [Leaderboard](leaderboard/主管护师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管护师资格考试),
Internal medicine head nurse: [Leaderboard](leaderboard/内科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内科主管护师),
Associate chief and chief nurse qualification exam: [Leaderboard](leaderboard/高级护师-副主任、主任护师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=高级护师-副主任、主任护师资格考试)
<br><br>


### 3.3 Pharmacists
☛☛ For the complete leaderboard, see [Pharmacists](leaderboard/药师.md)<br>
Licensed pharmacist (Western medicine): [Leaderboard](leaderboard/执业西药师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=执业西药师),
Licensed pharmacist (Traditional Chinese medicine): [Leaderboard](leaderboard/执业中药师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=执业中药师),
Junior pharmacist exam: [Leaderboard](leaderboard/药士初级考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=药士初级考试),
Junior pharmacist (pharmacist) exam: [Leaderboard](leaderboard/药师初级考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=药师初级考试),
Traditional Chinese pharmacy (junior): [Leaderboard](leaderboard/初级中药士-中药学（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级中药士-中药学（士）),
Traditional Chinese pharmacy (pharmacist): [Leaderboard](leaderboard/初级中药师-中药学（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级中药师-中药学（师）),
Chief pharmacist qualification exam: [Leaderboard](leaderboard/主管药师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管药师资格考试),
Chief pharmacist of traditional Chinese medicine: [Leaderboard](leaderboard/主管中药师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管中药师)
<br><br>


### 3.4 Medical Technology
☛☛ For the complete leaderboard, see [Medical Technology](leaderboard/医技.md)<br>
Ultrasound department: [Leaderboard](leaderboard/规培结业-超声科.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-超声科),
Attending physician of ultrasonic medicine: [Leaderboard](leaderboard/超声波医学主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=超声波医学主治医师),
Chief technician of ultrasonic medicine: [Leaderboard](leaderboard/超声波医学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=超声波医学主管技师),
Chief technician of electrocardiography: [Leaderboard](leaderboard/心电学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心电学主管技师),
Medical imaging department: [Leaderboard](leaderboard/规培结业-医学影像科.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-医学影像科),
Attending physician of nuclear medicine: [Leaderboard](leaderboard/核医学主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=核医学主治医师),
Chief technician of nuclear medicine: [Leaderboard](leaderboard/核医学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=核医学主管技师),
Attending physician of radiology: [Leaderboard](leaderboard/放射科主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射科主治医师),
Radiological technology (junior): [Leaderboard](leaderboard/放射学技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射学技术（士）),
Radiological technology (technician): [Leaderboard](leaderboard/放射学技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射学技术（师）),
Chief technician of radiological medicine: [Leaderboard](leaderboard/放射医学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射医学主管技师) ,
Laboratory technology (junior): [Leaderboard](leaderboard/检验技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=检验技术（士）),
Laboratory technology (technician): [Leaderboard](leaderboard/检验技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=检验技术（师）),
Chief technician of microbiological testing: [Leaderboard](leaderboard/微生物检验主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=微生物检验主管技师),
Chief technician of physicochemical testing: [Leaderboard](leaderboard/理化检验主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=理化检验主管技师),
Chief technician of clinical medical testing: [Leaderboard](leaderboard/临床医学检验主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学检验主管技师), 
Attending physician of pathology: [Leaderboard](leaderboard/病理科主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=病理科主治医师),
Chief technician of pathology: [Leaderboard](leaderboard/病理学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=病理学主管技师),
Pathological technology: [Leaderboard](leaderboard/主管技师-病理学技术.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管技师-病理学技术), 
Rehabilitation medicine therapy technology (junior): [Leaderboard](leaderboard/康复医学治疗技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学治疗技术（士）),
Rehabilitation medicine therapy technology (technician): [Leaderboard](leaderboard/康复医学治疗技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学治疗技术（师）),
Chief technician of rehabilitation medicine and therapy: [Leaderboard](leaderboard/康复医学与治疗主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学与治疗主管技师),
Oncology technology (junior): [Leaderboard](leaderboard/肿瘤学技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤学技术（士）),
Oncology technology (technician): [Leaderboard](leaderboard/肿瘤学技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤学技术（师）),
Chief technician of tumor radiotherapy: [Leaderboard](leaderboard/肿瘤放射治疗主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤放射治疗主管技师),
Chief technician of transfusion technology: [Leaderboard](leaderboard/输血技术主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=输血技术主管技师),
Chief technician of disinfection technology: [Leaderboard](leaderboard/消毒技术主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=消毒技术主管技师),
Chief technician of medical record information: [Leaderboard](leaderboard/病案信息主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=病案信息主管技师)
<br><br>


### 3.5 Basic Medical Knowledge
(1) Basic Medicine, [Leaderboard](leaderboard/基础医学.md)<br>
Medical "Three Basics": [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学三基),
Medical psychology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学心理学),
Biochemistry and molecular biology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=生物化学与分子生物学),
Cell biology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-细胞生物学),
Medical immunology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学免疫学),
Immunology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-免疫学),
Pathophysiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-病理生理学),  
Pathology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-病理学),
Medical genetics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学遗传学),
Parasitology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-寄生虫学),
Human parasitology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-人体寄生虫学),
Systemic anatomy: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-系统解剖学),
Anatomy: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-解剖学),
Regional anatomy: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-局部解剖学),
Bioinformatics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-生物信息学),
Physiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-生理学),
Pharmacology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-药理学),
Pharmaceutical analysis: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-药物分析学),
Medical microbiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学微生物学),
Histology and embryology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-组织学与胚胎学),
Medical statistics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学统计学)
<br>

(2) Clinical Medicine, [Leaderboard](leaderboard/临床医学.md)<br>
Clinical medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学综合),
Medical imaging: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-医学影像学),
Radiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-放射学),
Laboratory diagnostics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-实验诊断学),
Neurology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-神经病学),
Surgery: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-外科学),
Dermatology and venereology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-皮肤性病学),
Pediatrics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-儿科学),
Nuclear medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-核医学),
Physical diagnostics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-物理诊断学),
Endodontics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-牙体牙髓病学),
Fundamentals of nursing: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-护理学基础),
Nursing: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-护理学基础),
Basic nursing: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-基础护理学),
Diagnostics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-诊断学),
Ultrasound medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-超声医学),
Oral nursing: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-口腔护理学),
Evidence-based medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-循证医学),
Epidemiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-流行病学),
Oral histopathology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-口腔组织病理学),
Infectious diseases: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-传染病学),
Oral anatomy and physiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-口腔解剖生理学),
Anesthesiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-麻醉学),
Interventional radiology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-介入放射学)
<br>

(3) Preventive Medicine and Public Health, [Leaderboard](leaderboard/预防医学与公共卫生学.md)<br>
Preventive medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=预防医学),
Hygiene: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=卫生学),
Medical ethics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学伦理学)
<br>

(4) Traditional Chinese Medicine and Chinese Pharmacy, [Leaderboard](leaderboard/中医学与中药学.md)<br>
Traditional Chinese ophthalmology: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医眼科学),
Jingui Yaolue Lectures: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金匮要略讲义),
Basic theory of traditional Chinese medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医基础理论),
Diagnostics of traditional Chinese medicine: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医诊断学),
Traditional
## 4. Financial Leaderboard
☛☛See the complete leaderboard at [Finance](leaderboard/金融.md)<br>

### 4.1 Accounting
☛☛See the complete leaderboard at [Accounting](leaderboard/财务.md).<br>
Junior Accountant Qualification: [Leaderboard](leaderboard/初级会计职称.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级会计职称),
Certified Public Accountant: [Leaderboard](leaderboard/注册会计师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=注册会计师),
Accounting Practitioner Qualification: [Leaderboard](leaderboard/会计从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=会计从业资格),
Auditor Exam: [Leaderboard](leaderboard/审计师考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=审计师考试),
Certified Tax Agent: [Leaderboard](leaderboard/注册税务师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=注册税务师),
Certified Management Accountant: [Leaderboard](leaderboard/注册管理会计师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=注册管理会计师)

### 4.2 Banking
☛☛See the complete leaderboard at [Banking](leaderboard/银行.md).<br>
Junior Banking Qualification: [Leaderboard](leaderboard/银行初级资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=银行初级资格),
Intermediate Banking Qualification: [Leaderboard](leaderboard/银从中级资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=银从中级资格),
Banking Practitioner Qualification: [Leaderboard](leaderboard/银行从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=银行从业资格)

### 4.3 Insurance
☛☛See the complete leaderboard at [Insurance](leaderboard/保险.md).<br>
Insurance Practitioner Qualification: [Leaderboard](leaderboard/保险从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险从业资格)

### 4.4 Securities
☛☛See the complete leaderboard at [Securities](leaderboard/证券.md).<br>
Securities Special Exam: [Leaderboard](leaderboard/证券专项考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=证券专项考试),
Securities Practitioner Qualification: [Leaderboard](leaderboard/证券从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=证券从业资格)

### 4.5 Other Financial Qualification Exams
☛☛See the complete leaderboard at [Other Financial Qualification Exams](leaderboard/其他金融资格考试.md).<br>
Junior Economist: [Leaderboard](leaderboard/初级经济师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级经济师),
Intermediate Economist: [Leaderboard](leaderboard/中级经济师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中级经济师),
Anti-Counterfeit Currency Knowledge: [Leaderboard](leaderboard/反假货币知识.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=反假货币知识),
Futures Practitioner Qualification: [Leaderboard](leaderboard/期货从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=期货从业资格),
AFP Financial Planner: [Leaderboard](leaderboard/金融理财师AFP.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融理财师AFP),
Fund Practitioner Qualification: [Leaderboard](leaderboard/基金从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基金从业资格),
Gold Practitioner Qualification: [Leaderboard](leaderboard/黄金从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=黄金从业资格),
China Actuary: [Leaderboard](leaderboard/中国精算师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中国精算师)

### 4.6 Financial Fundamentals
☛☛See the complete leaderboard at [Financial Fundamentals](leaderboard/金融基础知识.md).<br>
Finance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融学),
Corporate Strategy and Risk Management: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公司战略与风险管理),
Macroeconomics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=宏观经济学),
Financial Markets: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融市场学),
Accounting: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=会计学),
Cost Accounting: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=成本会计学),
Monetary Finance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=货币金融学),
Political Economics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=政治经济学),
Investment: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=投资学),
Econometrics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=计量经济学),
Corporate Finance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公司金融学),
Public Finance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=财政学),
Commercial Bank Finance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=商业银行金融学),
Management Accounting: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=管理会计学),
Central Banking: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中央银行学),
Auditing: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=审计学),
International Economics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=国际经济学),
Intermediate Financial Accounting: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中级财务会计),
Financial Management: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=财务管理学),
Microeconomics: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=微观经济学),
International Finance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=国际金融学),
Financial Engineering: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融工程学),
Economic Law: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=经济法),
Advanced Financial Accounting: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=高级财务会计),
Insurance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险学)

### 4.7 Financial Applications
☛☛See the complete leaderboard at [Financial Applications](leaderboard/金融应用.md).<br>
Insurance Knowledge Interpretation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险知识解读),
Financial Terminology Explanation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融术语解释),
Practicing Physician Qualification Exam: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融知识-执业医师资格考试),
Wealth Management Knowledge Interpretation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=理财知识解读),
Licensed Pharmacist Qualification Exam: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融知识-执业药师资格考试),
Financial Document Extraction: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融文档抽取),
Research Viewpoint Extraction: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融认知-研判观点提取),
Financial Sentiment Recognition: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融情绪识别),
Insurance Slot Recognition: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险槽位识别),
Insurance Intent Understanding: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险意图理解),
Financial Intent Understanding: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融意图理解),
Insurance Attribute Extraction: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险属性抽取),
Insurance Clause Interpretation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险条款解读),
Financial Product Analysis: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融产品分析),
Financial Numerical Calculation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融数值计算),
Financial Event Interpretation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融事件解读),
Content Generation - Investor Education Script Generation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融投教话术生成),
Content Generation - Text Summarization: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融文本总结归纳),
Content Generation - Marketing Copy Generation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融营销文案生成),
Content Generation - News Headline Generation: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融资讯标题生成),
Security Compliance - Financial Compliance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融合规性),
Security Compliance - Financial Issue Identification: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融问题识别),
Security Compliance - Information Security Compliance: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融信息安全合规),
Security Compliance - Financial Factuality: [badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融事实性)
<br><br><br>

## 5. Law and Civil Service Rankings
☛☛See the complete ranking at [Law and Civil Service](leaderboard/法律与行政公务.md)<br>

### 5.1 Lawyer Qualification Exam
#### (1) JEC-QA-KD
Multiple-choice questions, 1000 in total, based on [AGIEval](https://github.com/ruixiangcui/AGIEval).<br>
See the complete ranking at [JEC-QA-KD](leaderboard/JEC-QA-KD.md), ☛View [JEC-QA-KD: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=JEC-QA-KD)
<br>

#### (2) JEC-QA-CA
Multiple-choice questions, 1000 in total, based on [AGIEval](https://github.com/ruixiangcui/AGIEval).<br>
See the complete ranking at [JEC-QA-CA](leaderboard/JEC-QA-CA.md), ☛View [JEC-QA-CA: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=JEC-QA-CA)
<br>

#### (3) Comprehensive Law
See the complete ranking at [Comprehensive Law](leaderboard/法律综合.md), ☛View [Comprehensive Law: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=法律综合)
<br><br><br>


### 5.2 Civil Service Exam
Civil service exam multiple-choice questions, 651 in total, based on [AGIEval](https://github.com/ruixiangcui/AGIEval).
Sample evaluation question:
> A township is planning a new district and decides to build a themed community in each of the east, south, west, and north directions, centered around a city park. The four communities are designated as the cultural district, leisure district, commercial district, and administrative service district. It is known that the administrative service district is southwest of the cultural district, and the cultural district is southeast of the leisure district.   
Based on the above statements, which of the following can be concluded?   
(A) The city park is north of the administrative service district (B) The leisure district is southwest of the cultural district (C) The cultural district is northeast of the commercial district (D) The commercial district is southeast of the leisure district   
>  

See the complete ranking at [Civil Service Exam](leaderboard/考公.md)<br>
☛View [Civil Service Exam: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=kaogong-so)
<br><br><br>



## 6. Reasoning and Mathematical Calculation Rankings
☛☛See the complete ranking at [Reasoning and Mathematical Calculation](leaderboard/推理与数学计算.md)<br>

### 6.1 Deductive Reasoning
Deductive reasoning (modus_tollens) multiple-choice questions, 123 in total, based on [ISP](https://arxiv.org/abs/2306.09479).

Sample evaluation question:
> Consider the following statements:  
1. If John is a good parent, then John is strict but fair. 2. John is not strict but fair. Conclusion: Therefore, John is not a good parent.
Question: Based on statements 1 and 2, is the conclusion correct?   
Answer: (A) No   (B) Yes   
>  

See the complete ranking at [Deductive Reasoning](leaderboard/演绎推理.md)<br>
☛View [Deductive Reasoning: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=演绎推理)
<br><br>


### 6.2 Commonsense Reasoning
Commonsense reasoning multiple-choice questions, 99 in total, based on [ISP](https://arxiv.org/abs/2306.09479).

Sample evaluation question:
> The following is a multiple-choice question about common sense.   
Question: When someone puts a potato into the embers beside a campfire, at this time the embers are not   
A. releasing heat  B. absorbing heat   
>      

See the complete ranking at [Commonsense Reasoning](leaderboard/常识推理.md)<br>
☛View [Commonsense Reasoning: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=常识推理)
<br><br>


### 6.3 Symbolic Reasoning BBH
The most commonly used symbolic reasoning evaluation set in academia, containing 23 sub-tasks. For details, see [BBH](https://nonelinear.com/static/benchmarks.html).
Sample evaluation question:
> Task description: Answer questions about which times certain events could have occurred.  
Q: Today, Emily went to the museum. Between what times could they have gone?   
We know that:   
Emily woke up at 1pm.   
Elizabeth saw Emily reading at the library from 2pm to 4pm.   
Jessica saw Emily watching a movie at the theater from 4pm to 5pm.    
Leslie saw Emily waiting at the airport from 5pm to 6pm.   
William saw Emily buying clothes at the mall from 6pm to 7pm.   
The museum was closed after 7pm.   
Between what times could Emily have gone to the museum?   
Options:   
(A) 1pm to 2pm   (B) 6pm to 7pm   (C) 5pm to 6pm   (D) 2pm to 4pm   
A:  
> 

See the complete ranking at [BBH](leaderboard/bbh.md)<br>
☛View [BBH Symbolic Reasoning: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=BBH)
<br><br>


### 6.4 Arithmetic Ability
Tests the basic arithmetic ability of large models, with questions involving integer addition and subtraction within 1000, and floating-point addition, subtraction, multiplication, and division with no more than 2 significant digits.
Example: 166 + 215 + 53 = ?, 0.97 + 0.4 / 4.51 = ?

See the complete ranking at [Arithmetic Ability](leaderboard/算术能力.md)<br>
☛View [Arithmetic Ability: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=算术能力)
<br><br>


### 6.5 Table Question Answering
Specifically tests a large model's ability to understand and analyze tables, commonly used in data analysis.    
Sample evaluation question:
> Name, Age, Gender, Nationality, Height (cm), Weight (kg), Education   
Zhang San, 28, Male, China, 180, 70, Bachelor's   
Lisa, 33, Female, USA, 165, 58, Master's   
Paulo, 41, Male, Brazil, 175, 80, PhD   
Miyuki, 25, Female, Japan, 160, 50, Associate's   
Ahmed, 30, Male, Egypt, 175, 68, Bachelor's   
Maria, 29, Female, Mexico, 170, 65, Master's   
Antonio, 36, Male, Spain, 182, 75, PhD  
Based on this table, answer: Which nationality has the lowest education level?
> 

See the complete ranking at [Table Question Answering](leaderboard/表格问答.md)<br>
☛View [Table Question Answering: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=表格问答)
<br><br>


### 6.6 Table Summarization
Specifically tests a large model's ability to analyze and summarize tables, commonly used in data analysis and article writing. There is no fixed standard answer, but it is relatively easy to objectively distinguish good from bad.
Sample evaluation question (part of the data omitted due to length):
> |Category|Organization|Large Model|Accuracy|Avg Time|Avg Token Usage|Cost/1k (Yuan)|Rank (Accuracy)|  
> |---|---|-----|-------------------|-------|-----------|-----------|-----------|  
> |Commercial|Doubao|doubao-seed-1-6-thinking-250715|87.5|37s|1976|14.6|1|   
> |Commercial|Baidu|ERNIE-4.5-Turbo-32K|84.7|33s|676|1.8|2|   
> |Commercial|Tencent|hunyuan-t1-20250711|84.7|37s|2465|9.2|3|   
> |Commercial|Tencent|hunyuan-turbos-20250716|83.9|24s|1288|2.3|4|   
> |...|...|...|...|...|...|...|...|   
> -------------------------   
> Given the new models: GLM-4.5, GLM-4.5-Air, GLM-4.5-Flash, step-3.   
> Based on the above table, write a summary in the following format: "xx organizations occupy the top 5 (do not repeat organization names), then describe the distribution of open-source and commercial models. Among the new models, xx ranks xx, xx ranks xx... (ranked from highest to lowest)." Strictly use the model and organization names as shown in the table.   
>   

See the complete ranking at [Table Summarization](leaderboard/表格总结.md)<br>
☛View [Table Summarization: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=表格总结)
<br><br>


### 6.7 High School Math Olympiad
2024 preliminary exam questions, based on [Math24o](https://github.com/CLUEbenchmark/Math24o).
Sample evaluation question:
> Let the set $S=\{1, 2, 3, \cdots, 997, 998 \}$, and let $A_{1},A_{2}, \cdots, A_{k}$ be $k$ subsets of $S$ each with 499 elements, such that for any two-element subset $B$ of $S$, there exists $i \in\{1, 2, \cdots, k \}$ such that $B \subset A_{i}$. Find the minimum value of $k$.
> 

See the complete ranking at [High School Math Olympiad](leaderboard/高中奥数.md)<br>
☛View [High School Math Olympiad: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=高中奥数)
<br><br>


### 6.8 Junior High Math Olympiad TODO
<br>


### 6.9 Elementary Math Olympiad
See the complete ranking at [Elementary Math Olympiad](leaderboard/小学奥数.md)<br>
☛View [Elementary Math Olympiad: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=小学奥数一年级)
<br><br>


### 6.10 Map Reasoning TODO
### 6.11 Spatial Reasoning TODO
<br>


### 6.12 Sudoku
See the complete ranking at [Sudoku](leaderboard/数独.md)<br>
☛View [Sudoku: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=数独入门)
<br>


### 6.13 Amount Uppercase Conversion TODO
### 6.14 Date Calculation TODO
<br><br><br>

## 7. Language and Instruction Following Leaderboard
☛☛See the full leaderboard at [Language and Instruction Following](leaderboard/语言与指令遵从.md)<br>

### 7.1 Idiom Understanding
Given a context, select the most appropriate idiom.

Sample evaluation:
> After discussing the strengths of the work, let's talk about why the ending is ____. The film itself raises sharp topics, and the "helping brother complex" has become an uncertain factor in many young people's marriages. So for something so sensitive, the film's ending simply uses the brother's cuteness to resolve the sister's issues, and finally chooses to stay and take care of...  
Choose the most suitable idiom or saying to fill in the blank above:  
(A) Organized and logical   (B) Biased listening   (C) Adding a dog's tail to a sable (meaning a poor ending)   (D) Half the country   (E) Life and property   (F) Timid as a mouse   (G) Maintaining one's own integrity  
> 

See the full leaderboard at [Idiom Understanding](leaderboard/成语理解.md)<br>
☛See [Idiom Understanding: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=成语理解)
<br><br>


### 7.2 Sentiment Analysis
Analyze the sentiment of user comments, negative or positive.

Sample evaluation:
> After using it for a few days, I found many problems. The Wi-Fi disconnects easily, the screen scratches easily, and opening web pages often crashes. Not worth buying.  
Is the above user comment positive or negative?  
(A) Negative   (B) Positive  
>    

See the full leaderboard at [Sentiment Analysis](leaderboard/情感分析.md)<br>
☛See [Sentiment Analysis: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=情感分析)
<br><br>


### 7.3 Textual Entailment
Textual entailment: determine the semantic relationship between two sentences: entailment, neutral, or contradiction. Reference: [OCNLI](https://arxiv.org/abs/2010.05444).

Sample evaluation:
> Sentence 1: The agricultural machinery purchase subsidy covers all agricultural and pastoral counties (farms) nationwide, and the central government plans to allocate 13 billion yuan, an increase of 9 billion yuan over last year.  
Sentence 2: Subsidies are distributed according to the number of farmers.  
What is the relationship between the two sentences above?  
(A) Entailment  (B) Neutral  (C) Contradiction  
>   

See the full leaderboard at [Textual Entailment](leaderboard/文本蕴含.md)<br>
☛See [Textual Entailment: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=文本蕴含)
<br><br>


### 7.4 Text Classification
Sample evaluation:
> Classify the following words by part of speech.  
> Dog, chase, run, adult, happy, tree

See the full leaderboard at [Text Classification](leaderboard/文本分类.md)<br>
☛See [Text Classification: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=文本分类)
<br><br>


### 7.5 Information Extraction
Sample evaluation:  
> "China CITIC Bank 300 million yuan, Bank of Communications increased by about 270 million yuan, China Everbright Bank about 100 million yuan."  
> Extract all organization names from the above text.

See the full leaderboard at [Information Extraction](leaderboard/信息抽取.md)<br>
☛See [Information Extraction: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=信息抽取)
<br><br>


### 7.6 Reading Comprehension
Reading comprehension is a type of matching ability, testing understanding of given information.
Depending on the type of information, it can be subdivided into: passage Q&A, table Q&A, dialogue Q&A, etc.  
Sample evaluation:
> Dentist: Okay, let's take a look at your teeth. Based on your description and our examination, you may have some gum disease, which is causing nerve irritation and sensitivity. In addition, these black spots may be cavities.  
Patient: Oh, really? What should I do?  
Dentist: Don't worry, we can make a treatment plan for you. We need to treat the gum disease first, then remove the cavities and fill the holes. During this process, we will ensure your comfort and use advanced technology and materials for the best results.  
Patient: Okay, thank you, doctor. So when can I start treatment?  
Dentist: Let's schedule an appointment for you. Your treatment will start in two days. In the meantime, please continue brushing, use dental floss, and avoid overly sweet and acidic foods and drinks.  
Patient: Okay, I will. Thank you again, doctor.  
Dentist: You're welcome. We will do our best to help you restore healthy teeth.  
Based on the above conversation, what dental problems were found during the examination?
> 

See the full leaderboard at [Reading Comprehension](leaderboard/阅读理解.md)<br>
☛See [Reading Comprehension: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=阅读理解)
<br><br>


### 7.7 Pronoun Resolution
Chinese coreference resolution task, reference: [CLUEWSC2020](https://github.com/CLUEbenchmark/CLUEWSC2020).
Sample evaluation:
> Shaoping still didn't know how to explain his brother-in-law's situation to his grandmother, so he just casually said, "He made a mistake, and they sent him to labor re-education!"  
In the above text, does the "he" in "he made a mistake" refer to Shaoping?  Options: (A) Yes   (B) No  
>    

See the full leaderboard at [Pronoun Resolution](leaderboard/代词理解.md)<br>
☛See [Pronoun Resolution: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=代词理解)
<br><br>


### 7.8 Poetry Matching
Chinese classical poetry matching: given a modern description of a classical Chinese poem, select from four candidate lines the one that semantically matches the modern description.
Correct options are constructed using parallel corpora of classical poetry and modern translations, and incorrect candidates are retrieved from ancient poetry corpora using similarity search.
Reference: [CCPM](https://github.com/THUNLP-AIPoet/CCPM).
Sample evaluation:
> The dim lamp goes out and is lit again.  
Which of the following lines best matches the above text:  
(A) The fisherman's lamp goes out and lights up again   (B) The dying lamp goes out and is lit again   (C) The dying lamp dims and lights up again   (D) The dying lamp goes out and lights up again  
>    

See the full leaderboard at [Poetry Matching](leaderboard/诗词匹配.md)<br>
☛See [Poetry Matching: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=诗词匹配)
<br><br>


### 7.9 Chinese Instruction Following
Based on Google's IFEval, translated and adapted to Chinese, featuring 9 categories and 25 types of instructions, as shown below:
![lin](pic/IFEval.jpg)

See the full leaderboard at [IFEval](leaderboard/中文指令遵从.md)<br>
☛See [Chinese Instruction Following: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中文指令遵从)
<br><br>


### 7.10 Chinese Character Glyph
See the full leaderboard at [Chinese Character Glyph](leaderboard/汉字字形.md)<br>
☛See [Chinese Character Glyph: badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=汉字字形)
<br><br>


### 7.11 Chinese Pinyin TODO
### 7.12 Find Wrong Characters TODO
### 7.13 Sentence Understanding TODO
### 7.14 Punctuation TODO
### 7.15 Simplified-Traditional Chinese Conversion TODO
### 7.16 Language Identification TODO
<br><br><br>


## 8. Agent and Tool Use Leaderboard
Calculate the average score of TAU and BFCL-V3.<br>
☛☛See the full leaderboard at [Agent and Tool Use Leaderboard](leaderboard/agent与工具调用.md)<br>

### 8.1 TAU
See the full leaderboard at [TAU](leaderboard/TAU.md)<br>
#### (1) TAU-airline
See the full leaderboard at [TAU-airline](leaderboard/TAU-airline.md)<br>

#### (2) TAU-retail
See the full leaderboard at [TAU-retail](leaderboard/TAU-retail.md)
<br><br>


### 8.2 BFCL-V3
BFCL-V3 is a tool use evaluation set released by UC Berkeley, pioneering multi-turn, multi-step function call scenarios. It evaluates real model interaction capabilities through API state validation and is currently one of the most authoritative benchmarks for large model tool use.
<br>See the full leaderboard at [BFCL-V3](leaderboard/BFCL-V3.md)
<br><br><br>



## 9. Coding Leaderboard
Evaluates large model programming ability. See the full leaderboard at [coding](leaderboard/coding.md)<br>

### 9.1 livecodebench
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) provides a comprehensive and contamination-free evaluation of large language models' (LLMs) programming abilities. Specifically, LiveCodeBench continuously collects new problems over time from three major competition platforms—LeetCode, AtCoder, and CodeForces.
<br>See the full leaderboard at [livecodebench](leaderboard/livecodebench.md)
<br><br>


### 9.2 Terminal-Bench-2.0
[Terminal-Bench](https://github.com/harbor-framework/terminal-bench-2) is a popular benchmark for evaluating the ability of agents and language models to perform valuable work in containerized environments. Test tasks include protein synthesis assembly, asynchronous code debugging, and security vulnerability repair, among others.
<br>See the full leaderboard at [Terminal-Bench-2.0](leaderboard/Terminal-Bench-2.0.md)
<br><br><br>
## 10. Integration of LMArena and AA Scores
Integration of our ReLE evaluation (Chinese) with LMArena (English) and Artificial Analysis (AA, English) leaderboard data.

| Large Model                                    | ReLE Evaluation (Chinese)   |    | AA-Intelligence (English)   | AA-Coding (English)   | AA-Math (English)   |    | LMArena-Text-overall (English)   | LMArena-Text-coding (English)   | LMArena-WebDev (English)   |
|:---------------------------------------|:-------------|:---|:----------------------|:----------------|:--------------|:---|:---------------------------|:--------------------------|:---------------------|
| gemini-3-pro-preview(new)              | 72.5         |    | 72.8                  | 62.3            | 95.7          |    | 1495                       | 1541                      | 1487                 |
| gpt-5.1-high(new)                      | 69.7         |    | 69.7                  | 57.5            | 94.0          |    | 1454                       | 1496                      | /                    |
| gpt-5.1-medium(new)                    | 69.3         |    | /                     | /               | /             |    | /                          | /                         | /                    |
| gpt-5-high                             | /            |    | 68.5                  | 52.7            | 94.3          |    | 1436                       | 1470                      | 1473                 |
| GPT-5 Codex (high)                     | /            |    | 68.5                  | 53.5            | 98.7          |    | /                          | /                         | /                    |
| kimi-k2-thinking(new)                  | 67.9         |    | 67.0                  | 52.2            | 94.7          |    | 1422                       | 1473                      | /                    |
| gpt-5-2025-08-07                       | 68.9         |    | 66.4                  | 49.2            | 91.7          |    | /                          | /                         | /                    |
| DeepSeek-V3.2-Think                    | 70.9         |    | 66.0                  | /               | /             |    | /                          | /                         | /                    |
| DeepSeek-V3.2                          | 64.4         |    | 52.0                  | /               | /             |    | /                          | /                         | /                    |
| o3                                     | /            |    | 65.5                  | 52.2            | 88.3          |    | 1435                       | 1458                      | 1186                 |
| grok-4-0709                            | 61.2         |    | 65.3                  | 55.1            | 92.7          |    | 1410                       | 1435                      | 1174                 |
| ...    | ...      |    | ...             | ...        | ...        |    | ...                  | ...                  | ...             |

For the complete scores, see [LMArena+AA](LMArena+AA.md)
<br><br>


## 🌐 Scores for Each Capability
Scoring method: Each large model is scored across various dimensions, with each dimension corresponding to an evaluation dataset containing several questions.
Each question is scored from 1 to 5 based on the quality of the model's response. The total score for all questions in the evaluation set is summed and normalized to a 100-point scale, which is used as the final score.

All scoring data can be found at [alldata](leaderboard/alldata.md)
<br><br>


## Why Create This Leaderboard?
- The large model landscape is flourishing, but quality varies greatly. Many media outlets tend to exaggerate and gloss over shortcomings, which can mislead the public; some companies, for PR purposes, also overstate their models' capabilities, frequently claiming to have "reached ChatGPT level" or to be "the best in China."
As the saying goes, "Laymen watch the excitement, experts see the subtleties." The industry urgently needs a breath of fresh air to abandon the hype, focus on refining cutting-edge technology, and let technical strength speak for itself. This requires an open, fair, and impartial evaluation system for large models, one that transparently displays the strengths and weaknesses of each model.
In this way, everyone can grasp the current state of development, understand the gap with top international technologies, and more clearly see the direction for future efforts, rather than being swept up by capital or media hype.
- For the industry, especially companies without large model R&D capabilities, understanding the technical boundaries of large models and efficiently making targeted technology selections is more important than ever.
An open, fair, and impartial evaluation system for large models can provide the necessary support, help avoid reinventing the wheel, prevent unnecessary disputes due to different tech stacks, and eliminate "talking past each other."
- For large model developers, enthusiasts, and academics who value practical results, comparing the effectiveness of various models reflects the validity of different technical approaches and methods, providing valuable reference.
Mutual reference and learning among different models helps everyone avoid unnecessary pitfalls, reduces resource waste from repeated experiments, and contributes to the healthy and efficient development of the entire large model ecosystem.
<br><br>


## Contact Us (Feixian Intelligence ReLE Benchmark Team)
### Large Model Evaluation Discussion Group
Add the editor on WeChat first, then you will be invited to the group. Please note "from github, join group"<br>
![lin](pic/qrcode-wxgroup.jpg)
<br><br><br>
### Large Model Evaluation Official WeChat Account
Follow the official WeChat account for large model evaluation to get the latest updates<br>
![lin](pic/qrcode-gzh.jpg)
<br><br><br>

---

## 📖 How to Cite ReLE Evaluation (Cite Us)

If you use ReLE (chinese-llm-benchmark) data, results, or code in your paper, report, or open-source project, please cite us using the following formats to help us maintain the open evaluation ecosystem.

### Chinese Citation (GB/T 7714)
ReLE Evaluation Group. ReLE: Chinese AI Large Model Capability Evaluation Dataset and Open Leaderboard [EB/OL]. GitHub, 2023-06-04[2025-12-06]. https://github.com/jeinlee1991/chinese-llm-benchmark. DOI: 10.5281/zenodo.xxxxxxx.

### APA (7th)
ReLE Benchmark Team. (2023, June 4). *ReLE: Really Reliable Live Evaluation for Chinese LLMs* (Version v5.8.5) [Computer software]. GitHub. https://github.com/jeinlee1991/chinese-llm-benchmark

### IEEE
[1] ReLE Benchmark Team, "ReLE: Really Reliable Live Evaluation for Chinese LLMs," GitHub repository, v5.8.5, Jun. 4, 2023. https://github.com/jeinlee1991/chinese-llm-benchmark

### BibTeX
```bibtex
@misc{rele2023benchmark,
  author       = {{ReLE Benchmark Team}},
  title        = {ReLE: Really Reliable Live Evaluation for Chinese LLMs},
  year         = {2025},
  url          = {https://github.com/jeinlee1991/chinese-llm-benchmark},
  version      = {v5.8.5},
  publisher    = {GitHub}
}
```

### Version Number Explanation
ReLE uses semantic versioning (`major.minor.patch`).  
- Major version: Major framework or metric weighting adjustments  
- Minor version: Addition of new domains, sub-leaderboards, or >10% question bank expansion  
- Patch: Bug fixes, sample denoising, model additions  

Please indicate the **exact tag** you used (e.g., `v5.8.5`) when citing, to ensure results are reproducible.