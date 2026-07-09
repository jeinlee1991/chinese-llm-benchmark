
# 非线智能 NoneLinear - ReLE评测：中文AI大模型能力评测（持续更新）
- ReLE （**R**eally R**e**liable **L**ive **E**valuation for LLM），原名CLiB
- 目前已囊括391个大模型，覆盖chatgpt、gpt-5.5、谷歌gemini-3.1-pro、Claude-5、文心ERNIE-X1.1、ERNIE-5.1、qwen3.7-max、qwen3.7-plus、百川、讯飞星火、商汤senseChat等商用模型，
以及hy3、step3.7-flash、kimi-k2.7、ernie4.5、MiniMax-M3、deepseek-v4、Qwen3.6、llama4、智谱GLM-5.2、MiMo-V2、LongCat、gemma4、mistral等开源大模型。
- 支持多维度能力评测，包括Education、Healthcare & Mental Health、Finance、Law & Public Administration、推理与Mathematics计算、Language & Instruction Following、Agent & Tool Use等7个领域，以及细分的~300个维度（比如牙科、高中Chinese Language…）。See 我们的技术报告[ReLE: A Scalable System and Structured Benchmark for Diagnosing Capability Anisotropy in Chinese LLMs](https://www.arxiv.org/abs/2601.17399) Media coverage (Jiqizhixin):[全球304个中文大模型实测：没有“全能王者”，ReLE凭70%降本方案破解评估困局](https://www.jiqizhixin.com/articles/2026-02-03)
- 不仅提供Leaderboard，也提供规模**超200万的大模型缺陷库**！方便广大社区研究分析、改进大模型。
- 为您的私有大模型提供免费评测服务，联系我们(非线智能 ReLE benchmark团队)：[加微信](#联系我们非线智能-ReLE-benchmark团队)


# 目录
- [🔄最近更新](#最近更新)
- [⚓Popular GitHub LLM Evaluation Projects](#GitHub热门大模型评测项目)
- [📝Basic Model Information](#大模型基本信息)
- [📊Leaderboard](#-排行榜)
  - [0、多模态Leaderboard](#0多模态排行榜)
  - [1、Overall AbilityLeaderboard](#1综合能力排行榜)
    - [1.1 推理类模型Leaderboard](#11推理类模型排行榜)
    - [1.2 商用大模型Leaderboard（含开源模型的付费API）](#12商用大模型排行榜含开源模型的付费API)
    - [1.3 开源大模型Leaderboard](#13开源大模型排行榜)
  - [2、EducationLeaderboard](#2教育排行榜)
    - [2.1 Primary School Subjects](#21-小学学科) &nbsp;|&nbsp; [2.2 Middle School Subjects](#22-初中学科) &nbsp;|&nbsp; [2.3 Zhongkao (High School Entrance Exam) TODO](#23-中考TODO)
    - [2.4 High School Subjects](#24-高中学科) &nbsp;|&nbsp; [2.5 Gaokao (National College Entrance Exam)](#25-高考) &nbsp;|&nbsp; [2.6 Higher Education TODO](#26-高等教育TODO)
    - [2.7 Postgraduate Entrance Exam TODO](#27-考研TODO) &nbsp;|&nbsp; [2.8 Teacher Qualification TODO](#28-教师资格TODO)
  - [3、Healthcare & Mental HealthLeaderboard](#3医疗与心理健康排行榜)    
    - [3.1 Physicians](#31-医师) &nbsp;|&nbsp; [3.2 Nursing](#32-护理) &nbsp;|&nbsp; [3.3 Pharmacists](#33-药师)
    - [3.4 Medical Technologists](#34-医技) &nbsp;|&nbsp; [3.5 Basic Medical Knowledge](#35-医学基础知识) &nbsp;|&nbsp; [3.6 Medical Postgraduate Entrance Exam](#36-医学考研)
    - [3.7 Mental Health](#37-心理健康)
  - [4、FinanceLeaderboard](#4金融排行榜)
    - [4.1 Finance & Accounting](#41-财务) &nbsp;|&nbsp; [4.2 Banking](#42-银行) &nbsp;|&nbsp; [4.3 Insurance](#43-保险)
    - [4.4 Securities](#44-证券) &nbsp;|&nbsp; [4.5 Other Financial Qualification Exams](#45-其他金融资格考试) &nbsp;|&nbsp; [4.6 Basic Financial Knowledge](#46-金融基础知识)
    - [4.7 Financial Applications](#47-金融应用)
  - [5、Law & Public AdministrationLeaderboard](#5法律与行政公务排行榜)
    - [5.1 Bar Exam (Legal Qualification)](#51-律师资格考试)
    - [5.2 Civil Service Exam](#52-公务员考试)
  - [6、Reasoning & Mathematics Leaderboard](#6推理与数学计算排行榜)
    - [6.1 Deductive Reasoning](#61-演绎推理)  &nbsp;|&nbsp; [6.2 Commonsense Reasoning](#62-常识推理) &nbsp;|&nbsp; [6.3 Symbolic Reasoning (BBH)](#63-符号推理BBH)
    - [6.4 Arithmetic Ability](#64-算术能力) &nbsp;|&nbsp; [6.5 Table Q&A](#65-表格问答) &nbsp;|&nbsp; [6.6 Table Summarization](#66-表格总结)
    - [6.7 High School Olympiad Mathematics](#67-高中奥数) &nbsp;|&nbsp; [6.8 Middle School Olympiad Mathematics TODO](#68-初中奥数TODO) &nbsp;|&nbsp; [6.9 Primary School Olympiad Mathematics](#69-小学奥数)
    - [6.10 Map Reasoning TODO](#610-地图推理TODO) &nbsp;|&nbsp; [6.11 Spatial Reasoning TODO](#611-空间推理TODO) &nbsp;|&nbsp; [6.12 Sudoku](#612-数独)
    - [6.13 Currency Amount Numeral Conversion TODO](#613-金额大小写转换TODO) &nbsp;|&nbsp; [6.14 Date Calculation TODO](#614-日期计算TODO)
  - [7、Language & Instruction FollowingLeaderboard](#7语言与指令遵从排行榜)
    - [7.1 Idiom Comprehension](#71-成语理解) &nbsp;|&nbsp; [7.2 Sentiment Analysis](#72-情感分析) &nbsp;|&nbsp; [7.3 Textual Entailment](#73-文本蕴含) 
    - [7.4 Text Classification](#74-文本分类) &nbsp;|&nbsp; [7.5 Information Extraction](#75-信息抽取) &nbsp;|&nbsp; [7.6 Reading Comprehension](#76-阅读理解) 
    - [7.7 Pronoun Resolution](#77-代词理解) &nbsp;|&nbsp; [7.8 Classical Poetry Matching](#78-诗词匹配) &nbsp;|&nbsp; [7.9 Chinese Instruction Following](#79-中文指令遵从) 
    - [7.10 Chinese Character Glyphs](#710-汉字字形) &nbsp;|&nbsp; [7.11 Hanyu Pinyin TODO](#711-汉语拼音TODO) &nbsp;|&nbsp; [7.12 Typo Detection TODO](#712-找错别字TODO) 
    - [7.13 Sentence Comprehension TODO](#713-句子理解TODO) &nbsp;|&nbsp; [7.14 Punctuation TODO](#714-标点符号TODO) &nbsp;|&nbsp; [7.15 Traditional/Simplified Chinese Conversion TODO](#715-汉字繁简转换TODO) 
    - [7.16 Language Identification TODO](#716-语种识别TODO)
  - [8、Agent & Tool UseLeaderboard](#8agent与工具调用排行榜)
    - [8.1 TAU](#81-TAU)
    - [8.2 BFCL-V3](#82-BFCL-V3)
  - [9、CodingLeaderboard](#9coding排行榜)
    - [9.1 livecodebench](#91-livecodebench)
    - [9.2 Terminal-Bench-2.0](#92-Terminal-Bench-20)  
  - [10、Integrating LMArena and AA Scores](#10整合LMArena和AA分数)    
- [🌐Scores by Ability](#🌐各项能力评分)
- [Why build this leaderboard?](#为什么做榜单)
- [Model Selection & Evaluation Discussion Group](#大模型评测交流群)
- [Cite Us](#如何引用-ReLE-评测Cite-Us)

# Recent Evaluation Updates
- [2026/7/9] v5.10.13 release
  - New models added：hy3
- [2026/7/2] v5.10.12 release
  - New models added：claude-sonnet-5-thinking
- [2026/6/27] v5.10.12 release
  - New models added：doubao-seed-2-1-pro-260628、doubao-seed-2-1-turbo-260628、doubao-seed-evolving
- [2026/6/18] v5.10.11 release
  - New models added：glm-5.2
- [2026/6/16] v5.10.10 release
  - New models added：kimi-k2.7-code
- [2026/6/2] v5.10.9 release
  - New models added：MiniMax-M3、qwen3.7-plus、step-3.7-flash、claude-opus-4.8-thinking
- [2026/5/30] v5.10.8 release
  - New models added：claude-opus-4.8
- [2026/5/23] v5.10.7 release
  - New models added：qwen3.7-max
- [2026/5/21] v5.10.6 release
  - New models added：gemini-3.5-flash
- [2026/5/13] v5.10.5 release
  - New models added：ernie-5.1
- [2026/5/1] v5.10.4 release
  - New models added：qwen3.6-27b
- [2026/4/25] v5.10.3 release
  - New models added：deepseek-v4-flash、deepseek-v4-pro、gpt-5.5
- [2026/4/23] v5.10.2 release
  - New models added：mimo-v2.5、mimo-v2.5-pro
- [2026/4/21] v5.10.1 release
  - New models added：qwen3.6-max-preview、kimi-k2.6
  - Updated model：更新kimi-k2.5评测结果（修复reasoning_content未传入tool call的调用错误），分数及排名有所变化
- [2026/4/18] v5.10 release，[2026/4/15] v5.9 release，[2026/4/8] v5.8.23 release，[2026/4/6] v5.8.22 release，[2026/4/3] v5.8.21 release，[2026/3/19] v5.8.20 release，[2026/3/18] v5.8.19 release，[2026/3/17] v5.8.18 release，[2026/3/5] v5.8.17 release，[2026/2/25] v5.8.16 release，[2026/2/20] v5.8.15 release，[2026/2/14] v5.8.14 release，[2026/2/9] v5.8.13 release，[2026/2/2] v5.8.12 release，[2026/1/27] v5.8.11 release，[2026/1/22] v5.8.10 release，[2025/12/24] v5.8.9 release，[2025/12/23] v5.8.8 release，[2025/12/18] v5.8.7 release，[2025/12/13] v5.8.6 release，[2025/12/6] v5.8.5 release，[2025/12/3] v5.8.4 release，[2025/11/3] v5.8 release，[2025/10/24] v5.7 release，[2025/10/13] v5.6 release，[2025/9/30] v5.5 release，[2025/9/22] v5.4 release，[2025/9/14] v5.3 release，[2025/9/10] v5.2 release，[2025/9/6] v5.1 release，[2025/9/1] v5.0 release，[2025/8/26]v4.13 release，[2025/8/20]v4.12 release，[2025/8/15]v4.11 release，[2025/8/10]v4.10 release，[2025/8/7]v4.9 release，[2025/8/1]v4.8 release，[2025/7/29]v4.7 release，[2025/7/26]v4.6 release，[2025/7/23]v4.5 release，[2025/7/17]v4.4 release，[2025/7/13]v4.3 release，[2025/7/12]v4.2 release，[2025/7/9]v4.1 release，[2025/7/2]v4.0 release，[2025/6/23]v3.33 release，[2025/6/18]v3.32 release，[2025/6/16]v3.31 release，[2025/6/13]v3.30 release，[2025/6/9]v3.29 release，[2025/6/4]v3.28 release，[2025/5/29]v3.27 release，[2025/5/23]v3.26 release，[2025/5/18]v3.25 release，[2025/5/15]v3.24 release，[2025/5/10]v3.23 release，[2025/5/5]v3.22 release，[2025/5/2]v3.21 release，[2025/4/30]v3.20 release，[2025/4/28]v3.19 release，[2025/4/22]v3.18 release，[2025/4/17]v3.17 release，[2025/4/9]v3.16 release，[2025/4/5]v3.15 release，[2025/4/3]v3.14 release，[2025/3/31]v3.13 release，[2025/3/29]v3.12 release，[2025/3/27]v3.11 release，[2025/3/25]v3.10 release，[2025/3/23]v3.9 release，[2025/3/21]v3.8 release，[2025/3/19]v3.7 release，[2025/3/17]v3.6 release，[2025/3/15]v3.5 release，[2025/3/13]v3.4 release，[2025/3/11]v3.3 release，[2025/3/10]v3.2 release，[2025/3/7]v3.1 release，[2025/3/4]v3.0 release，[2025/3/3]v2.22 release，[2025/2/28]v2.21 release，[2025/2/24]v2.20 release，[2025/2/22]v2.19 release，[2025/2/18]v2.18 release，[2025/2/14]v2.17 release，[2025/2/13]v2.16 release，[2025/2/12]v2.15 release，[2025/2/10]v2.14 release，[2025/1/29]v2.13 release，[2025/1/25]v2.12 release，[2025/1/23]v2.11 release，[2025/1/22]v2.10 release，[2025/1/20]v2.9 release，[2025/1/17]v2.8 release，[2025/1/7]v2.7 release
- 2024：[2024/12/28]v2.6 release，[2024/12/27]v2.5 release，[2024/12/25]v2.4 release, [2024/10/20]v2.3 release，[2024/9/29]v2.2 release，[2024/8/27]v2.1 release，[2024/8/7]v2.0 release，[2024/7/26]v1.21 release，[2024/7/15]v1.20 release，[2024/6/29]v1.19 release，[2024/6/2]v1.18 release，[2024/5/8]v1.17 release，[2024/4/13]v1.16 release，[2024/3/20]v1.15 release，[2024/2/28]v1.14 release，[2024/1/29]v1.13 release
- 2023：[2023/12/10]v1.12 release，[2023/11/22]v1.11 release，[2023/11/5]v1.10 release，[2023/10/11]v1.9 release，[2023/9/13]v1.8 release，[2023/8/29]v1.7 release，[2023/8/13]v1.6 release，[2023/7/26]v1.5 release， [2023/7/18]v1.4 release， [2023/7/2]v1.3 release， [2023/6/17]v1.2版， [2023/6/10]v1.1 release， [2023/6/4]v1 release

Full version update details：[CHANGELOG](CHANGELOG.md)
<br><br>


# Popular GitHub LLM Evaluation Projects
| repo                                                                               | star  | area   | about                                                                                                                                                                                                                                                                   |
|------------------------------------------------------------------------------------|-------|--------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [langfuse](https://github.com/langfuse/langfuse)                                   | 23.6k | Overseas     | Open source LLM engineering platform: LLM Observability, metrics, evals, prompt management, playground, datasets. Integrates with OpenTelemetry, Langchain, OpenAI SDK, LiteLLM, and more. 🍊YC W23                                                                     |
| [opik](https://github.com/comet-ml/opik)                                           | 18.4k | Overseas     | Debug, evaluate, and monitor your LLM applications, RAG systems, and agentic workflows with comprehensive tracing, automated evaluations, and production-ready dashboards.                                                                                              |
| [deepeval](https://github.com/confident-ai/deepeval)                      | 14.2k | Overseas     | The LLM Evaluation Framework                                                                                                                                                                                                                                            |
|……|……|……|……|
| [⭐chinese-llm-benchmark（我们）](https://github.com/jeinlee1991/chinese-llm-benchmark) | 5.7k  | **Domestic (China)** | ReLE中文大模型能力评测（持续更新） |                                                                                               |
|……|……|……|……|

See [hot50](GitHub热门评测repo.md)
<br><br>


# Basic Model Information
- [Weekly New Models](每周最新模型.md)
    - [6月15~6月21](每周最新模型.md#6月156月21)
    - [6月8~6月14](每周最新模型.md#6月86月14)
    - [6月1~6月7](每周最新模型.md#6月16月7)
    - [5月25~5月31](每周最新模型.md#5月255月31)
- For more informationSee [Model list](https://nonelinear.com/static/models.html)
<br><br>

# 🚀 大模型统一网关
Introducing a one-stop AI model marketplace 🛒，offering the most comprehensive selection of large models, keeping you always one step ahead.
- 🌐 Global models, all in one place：GPT-5.5、Gemini-3.1-Pro、Claude-4.7、DeepSeek-v4、Kimi-k2.5……
- ⚖️ Smart load balancing & high concurrency：We aggregate multiple top-tier providers and use intelligent routing for automatic load balancing. Say goodbye to annoying Rate Limit errors and easily handle any traffic surge!
- 🔀 Automatic failover：Is a single provider's API temporarily glitching? No problem! Our system switches seamlessly to a healthy backup channel in milliseconds, ensuring 99.9999% availability for your service and sparing your users the awkwardness of "service unavailable".
- 🛡️Online monitoring & smart model selection：Seamlessly integrates with online performance monitoring tools, closing the loop between model selection and evaluation. Let real data speak for itself, helping you easily find the best-performing, most cost-effective model.
[如何接入在线效果监测](https://nonelinear.com/static/online-eval.html)，[如何接入模型选型评测](https://nonelinear.com/static/task-create.html)
- 💰 Outstanding value for money！☛[View all models and pricing](https://nonelinear.com/static/models.html)
```
from openai import OpenAI
base_url = "https://api.nonelinear.com/v1"
api_key = "<your api key>" # 获取https://nonelinear.com/static/apikey.html
client = OpenAI(api_key=api_key, base_url=base_url)
client.chat.completions.create(
    model="<model id>", # 模型列表https://nonelinear.com/static/models.html
    messages=[{"role": "user", "content": "<your prompt>"}],
)
```
<br><br>


# 💥Model Selection: Cut Costs by up to 90%
Stop choosing a large model blindly🎉！Upload your own custom test data📊，In 5 minutes🔍find out which model performs best for your scenario🏆、and offers the best value💰！Choose the most suitable model and cut costs by up to 90%💥！[Try it>>](https://nonelinear.com/static/task-create.html)
![link](docs/modelSelection/img/task-result-html.png)
<video controls src="docs/modelSelection/img/modelsel.mp4"></video>

Examples：
- [WeChat article writing - table summarization](docs/modelSelection/微信文章撰写之表格总结.md)
- [MathML to LaTeX conversion](docs/modelSelection/MathML转LaTeX格式.md)
<br><br>


# 📊 Leaderboard
## 0、多模态Leaderboard
See detailed data[Multimodal Evaluation](README-多模态评测.md)<br>
<br><br>


## 1、Overall AbilityLeaderboard
“Overall Ability”scoring method：“Overall Ability”is“Domain-specific Ability”和“General Ability”a weighted score, with weights of 0.3 and 0.7 respectively; where“Domain-specific Ability”为“Education”、“Healthcare & Mental Health”、“Finance”、“Law & Public Administration”average score across 4 domains，“General Ability”为“推理与Mathematics计算”、“Language & Instruction Following”、“Agent & Tool Use”、“Coding” average score across 4 domains。
![link](pic/总分.png)

|类别|机构|大模型|【总分】准确率|平均耗时|平均消耗token|花费/千次（元）|排名（准确率）|
|---|---|-----|-------------------|-------|-----------|-----------|-----------|
|商用|阿里巴巴|qwen3.7-max(new)|76.9%|51s|2920|99.0|1|
|商用|豆包|doubao-seed-evolving(new)|75.5%|267s|10392|304.7|2|

   
See detailed data：[Overall AbilityLeaderboard](leaderboard/总分.md) | [General AbilityLeaderboard](leaderboard/通用能力.md) | [Domain-specific AbilityLeaderboard](leaderboard/专业能力.md)
<br><br> 

#### 1.1、推理模型Leaderboard
See[推理模型Leaderboard](leaderboard/reasonmodel.md)<br>
<br>
#### 1.2、商用大模型Leaderboard（含开源模型的付费API）
[输出价格5元及以上商用大模型](leaderboard/commerce1.md) | [输出价格1~5元商用大模型](leaderboard/commerce2.md) | [输出价格1元以下商用大模型](leaderboard/commerce3.md)<br>
DIY自定义维度筛选榜单：☛ [link](https://nonelinear.com/static/benchmarking.html) 
<br>
<br>
#### 1.3、开源大模型Leaderboard
[5B以下开源大模型](leaderboard/opensource1.md) | [5B~20B开源大模型](leaderboard/opensource2.md) | [20B以上开源大模型](leaderboard/opensource3.md)<br>
DIY自定义维度筛选榜单：☛[link](https://nonelinear.com/static/benchmarking.html)

<br><br>



## 2、EducationLeaderboard
☛☛See full leaderboard: [Education](leaderboard/教育.md)<br>

### 2.1 Primary School Subjects
☛☛See full leaderboard: [Primary School Subjects](leaderboard/小学学科.md)。<br>
Chinese Language：[Leaderboard](leaderboard/PrimarySchoolChinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolChinese)，
English：[Leaderboard](leaderboard/PrimarySchoolEnglish.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolEnglish)，
Mathematics：[Leaderboard](leaderboard/PrimarySchoolMathematics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolMathematics)，
Ethics & Rule of Law：[Leaderboard](leaderboard/PrimarySchoolEthics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolEthics)，
Science：[Leaderboard](leaderboard/PrimarySchoolScience.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=PrimarySchoolScience)
<br><br>


### 2.2 Middle School Subjects
☛☛See full leaderboard: [Middle School Subjects](leaderboard/初中学科.md)。<br>
Biology：[Leaderboard](leaderboard/MiddleSchoolBiology.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolBiology)，
Chemistry：[Leaderboard](leaderboard/MiddleSchoolChemistry.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolChemistry)，
Chinese Language：[Leaderboard](leaderboard/MiddleSchoolChinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolChinese)，
English：[Leaderboard](leaderboard/MiddleSchoolEnglish.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolEnglish)，
Geography：[Leaderboard](leaderboard/MiddleSchoolGeography.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolGeography)，
History：[Leaderboard](leaderboard/MiddleSchoolHistory.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolHistory)，
Mathematics：[Leaderboard](leaderboard/MiddleSchoolMathematics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolMathematics)，
Physics：[Leaderboard](leaderboard/MiddleSchoolPhysics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolPhysics)，
Politics：[Leaderboard](leaderboard/MiddleSchoolPolitics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=MiddleSchoolPolitics)
<br><br>


### 2.3 Zhongkao (High School Entrance Exam) TODO

### 2.4 High School Subjects
☛☛See full leaderboard: [High School Subjects](leaderboard/高中学科.md)。<br>
Biology：[Leaderboard](leaderboard/HighSchoolBiology.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolBiology)，
Chemistry：[Leaderboard](leaderboard/HighSchoolChemistry.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolChemistry)，
Chinese Language：[Leaderboard](leaderboard/HighSchoolChinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolChinese)，
English：[Leaderboard](leaderboard/HighSchoolEnglish.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolEnglish)，
Geography：[Leaderboard](leaderboard/HighSchoolGeography.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolGeography)，
History：[Leaderboard](leaderboard/HighSchoolHistory.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolHistory)，
Mathematics：[Leaderboard](leaderboard/HighSchoolMathematics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolMathematics)，
Physics：[Leaderboard](leaderboard/HighSchoolPhysics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolPhysics)，
Politics：[Leaderboard](leaderboard/HighSchoolPolitics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=HighSchoolPolitics)
<br><br>


### 2.5 Gaokao (National College Entrance Exam)
历年Gaokao (National College Entrance Exam)真题，含简单题、填空题、选择题等等，只保留客观题。所有分数均为准确率，全部答对为100%；比如Mathematics100，表示全部答对。☛☛See full leaderboard: [Gaokao (National College Entrance Exam)](leaderboard/高考.md)。<br>
（1）2025年Gaokao (National College Entrance Exam)<br>
Biology：[Leaderboard](leaderboard/2025高考生物.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考生物)，
Chemistry：[Leaderboard](leaderboard/2025高考化学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考化学)，
Chinese Language：[Leaderboard](leaderboard/2025高考语文.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考语文)，
English：[Leaderboard](leaderboard/2025高考英语.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考英语)，
Geography：[Leaderboard](leaderboard/2025高考地理.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考地理)，
History：[Leaderboard](leaderboard/2025高考历史.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考历史)，
Mathematics：[Leaderboard](leaderboard/2025高考数学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考数学)，
Physics：[Leaderboard](leaderboard/2025高考物理.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考物理)，
Politics：[Leaderboard](leaderboard/2025高考政治.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=2025高考政治)。

（2）2024及之前Gaokao (National College Entrance Exam)<br>
Biology：[Leaderboard](leaderboard/gaokao-biology.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-biology)，
Chemistry：[Leaderboard](leaderboard/gaokao-chemistry.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-chemistry)，
Chinese Language：[Leaderboard](leaderboard/gaokao-chinese.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-chinese)，
Geography：[Leaderboard](leaderboard/gaokao-geography.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-geography)，
History：[Leaderboard](leaderboard/gaokao-history.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-history)，
Mathematics：[Leaderboard](leaderboard/gaokao-math.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-math)，
Physics：[Leaderboard](leaderboard/gaokao-physics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-physics)，
Politics：[Leaderboard](leaderboard/gaokao-politics.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-politics)。
<br><br>


### 2.6 Higher Education TODO
### 2.7 Postgraduate Entrance Exam TODO
### 2.8 Teacher Qualification TODO
<br><br><br>



## 3、Healthcare & Mental HealthLeaderboard
☛☛See full leaderboard: [Healthcare & Mental Health](leaderboard/医疗与心理健康.md)<br>

### 3.1 Physicians
☛☛See full leaderboard: [Physicians](leaderboard/医师.md)<br>
（1）内科，[Leaderboard](leaderboard/内科.md)<br>
Internal Medicine Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-内科)，
TCM Internal Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医内科主治医师)，
Internal Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内科主治医师)，
Cardiology & Respiratory Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心血管内科与呼吸内科主治医师)，
Nephrology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肾内科主治医师)，
Gastroenterology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=消化内科主治医师)，
Integrated Chinese-Western Medicine Internal Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中西医结合内科主治医师)，
Gastroenterology Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=消化内科高级职称)，
General Internal Medicine Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=普通内科高级职称)，
Respiratory Medicine Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=呼吸内科高级职称)，
Cardiology Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心内科高级职称)，
Tuberculosis Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=结核病主治医师)，
Endocrinology Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内分泌科高级职称)
<br>

（2）外科，[Leaderboard](leaderboard/外科.md)<br>
Surgery Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-外科)，
Oral & Maxillofacial Surgery Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔颌面外科主治医师)，
Plastic Surgery Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=整形外科主治医师)，
Surgery Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=外科主治医师)，
General Surgery Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=普通外科高级职称)，
Orthopedics Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-骨科)，
Orthopedics Intermediate Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=骨科中级职称)，
Orthopedics Senior Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=骨科高级职称)
<br>

（3）妇产科，[Leaderboard](leaderboard/妇产科.md)<br>
Obstetrics & Gynecology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-妇产科)，
Obstetrics & Gynecology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=妇产科主治医师)，
妇产Science副主任、Chief Physician Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=妇产科学副主任、主任医师职称考试)
<br>

（4）儿科，[Leaderboard](leaderboard/儿科.md)<br>
Pediatrics Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-儿科)，
Pediatrics Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=儿科主治医师)，
Pediatric Surgery Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-小儿外科) 
<br>

（5）眼科，[Leaderboard](leaderboard/眼科.md)<br>
Ophthalmology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-眼科)，
Ophthalmology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=眼科主治医师)
<br>

（6）口腔科，[Leaderboard](leaderboard/口腔科.md)<br>
Stomatology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-口腔科)，
Dental Assistant Physician Licensing Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔执业助理医师)，
Dental Licensed Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔执业医师)，
Restorative Dentistry Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔内科主治医师)，
Stomatology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔科主治医师)，
Prosthodontics Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔修复科主治医师)，
Orthodontics Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=口腔正畸学主治医师)
<br>

（7）耳鼻咽喉科，[Leaderboard](leaderboard/耳鼻咽喉科.md)<br>
ENT Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-耳鼻咽喉科)，
ENT Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=耳鼻咽喉科主治医师)
<br>

（8）脑系科，[Leaderboard](leaderboard/脑系科.md)<br>
Neurology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-神经内科)，
Neurology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=神经内科主治医师)，
Psychiatry Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-精神科)，
Psychiatry Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=精神病学主治医师)，
Psychotherapy Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理治疗学主治医师考试)，
Psychological Counsellor Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理咨询师考试)
<br>

（9）皮肤科，[Leaderboard](leaderboard/皮肤科.md)<br>
Dermatology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-皮肤科)，
Dermatology Intermediate Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=皮肤科中级职称)，
Dermatology & Venereology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=皮肤与性病学主治医师)
<br>

（10）中医与中西医结合，[Leaderboard](leaderboard/中医与中西医结合.md)<br>
Integrated Chinese-Western Medicine Assistant Physician Licensing Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中西医结合执业助理医师)，
TCM Assistant Physician Licensing Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医执业助理医师)，
Integrated Chinese-Western Medicine Licensed Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中西医结合执业医师)，
TCM Licensed Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医执业医师)，
TCM Acupuncture Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医针灸主治医师)
<br>

（11）康复医学科，[Leaderboard](leaderboard/康复医学科.md)<br>
Rehabilitation Medicine Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-康复医学科)，
Rehabilitation Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学主治医师)
<br>

（12）全科医学科，[Leaderboard](leaderboard/全科医学科.md)<br>
General Practice Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-全科医学科)，
General Practice Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=全科主治医师)
<br>

（13）临床营养与重症医学，[Leaderboard](leaderboard/临床营养与重症医学.md)<br>
Clinical Assistant Physician Licensing Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床执业助理医师)，
Clinical Licensed Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床执业医师)，
Rheumatology & Clinical Immunology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=风湿与临床免疫主治医师)，
Critical Care Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=重症医学主治医师)，
Nutrition Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=营养学主治医师)，
Clinical Pathology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-临床病理科)
<br>

（14）肿瘤科，[Leaderboard](leaderboard/肿瘤科.md)<br>
Oncology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤学主治医师)
<br>

（15）麻醉疼痛科，[Leaderboard](leaderboard/麻醉疼痛科.md)<br>
Anesthesiology Residency Completion Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-麻醉科)，
Anesthesiology Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=麻醉科主治医师)，
Pain Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=疼痛科主治医师)
<br>

（16）公共卫生与职业病，[Leaderboard](leaderboard/公共卫生与职业病.md)<br>
Public Health Assistant Physician Licensing Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公共卫生执业助理医师)，
Public Health Licensed Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公共卫生执业医师)，
Hospital Infection Control Intermediate Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医院感染中级职称)，
Infectious Disease Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=传染病主治医师)，
Preventive Medicine Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=预防医学主治医师)，
Infectious Diseases Intermediate Title Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=传染病学中级职称)，
Occupational Disease Attending Physician Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=职业病主治医师)
<br><br>


### 3.2 Nursing
☛☛See full leaderboard: [Nursing](leaderboard/护理.md)<br>
Registered Nurse Licensing Exam：[Leaderboard](leaderboard/护士执业资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=护士执业资格考试)，
Nurse Practitioner Qualification Exam：[Leaderboard](leaderboard/护师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=护师资格考试)，
Pediatrics Charge Nurse Exam：[Leaderboard](leaderboard/儿科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=儿科主管护师)，
Internal Medicine Nursing：[Leaderboard](leaderboard/主管护师-内科护理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管护师-内科护理学)，
Obstetrics & Gynecology Nursing：[Leaderboard](leaderboard/主管护师-妇产科护理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管护师-妇产科护理学)，
Obstetrics & Gynecology Charge Nurse Exam：[Leaderboard](leaderboard/妇产科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=妇产科主管护师)，
Surgery Charge Nurse Exam：[Leaderboard](leaderboard/外科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=外科主管护师)，
Charge Nurse Qualification Exam：[Leaderboard](leaderboard/主管护师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管护师资格考试)，
Internal Medicine Charge Nurse Exam：[Leaderboard](leaderboard/内科主管护师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内科主管护师)，
副主任、Chief Nurse Qualification Exam：[Leaderboard](leaderboard/高级护师-副主任、主任护师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=高级护师-副主任、主任护师资格考试)
<br><br>


### 3.3 Pharmacists
☛☛See full leaderboard: [Pharmacists](leaderboard/药师.md)<br>
Licensed Western Pharmacist Exam：[Leaderboard](leaderboard/执业西药师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=执业西药师)，
Licensed Chinese Pharmacist Exam：[Leaderboard](leaderboard/执业中药师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=执业中药师)，
Junior Pharmacy Technician Exam：[Leaderboard](leaderboard/药士初级考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=药士初级考试)，
Junior Pharmacist Exam：[Leaderboard](leaderboard/药师初级考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=药师初级考试)，
Chinese Pharmacy (Junior)：[Leaderboard](leaderboard/初级中药士-中药学（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级中药士-中药学（士）)，
Chinese Pharmacy (Pharmacist)：[Leaderboard](leaderboard/初级中药师-中药学（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级中药师-中药学（师）)，
Pharmacist-in-Charge Qualification Exam：[Leaderboard](leaderboard/主管药师资格考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管药师资格考试)，
Chinese Pharmacist-in-Charge Exam：[Leaderboard](leaderboard/主管中药师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管中药师)
<br><br>


### 3.4 Medical Technologists
☛☛See full leaderboard: [Medical Technologists](leaderboard/医技.md)<br>
Ultrasound Department：[Leaderboard](leaderboard/规培结业-超声科.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-超声科)，
Ultrasound Medicine Attending Physician Exam：[Leaderboard](leaderboard/超声波医学主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=超声波医学主治医师)，
Ultrasound Medicine Chief Technician Exam：[Leaderboard](leaderboard/超声波医学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=超声波医学主管技师)，
Electrocardiography Chief Technician Exam：[Leaderboard](leaderboard/心电学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心电学主管技师)，
Medical Imaging Department：[Leaderboard](leaderboard/规培结业-医学影像科.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=规培结业-医学影像科)，
Nuclear Medicine Attending Physician Exam：[Leaderboard](leaderboard/核医学主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=核医学主治医师)，
Nuclear Medicine Chief Technician Exam：[Leaderboard](leaderboard/核医学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=核医学主管技师)，
Radiology Attending Physician Exam：[Leaderboard](leaderboard/放射科主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射科主治医师)，
Radiologic Technology (Junior)：[Leaderboard](leaderboard/放射学技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射学技术（士）)，
Radiologic Technology (Technician)：[Leaderboard](leaderboard/放射学技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射学技术（师）)，
Radiology Chief Technician Exam：[Leaderboard](leaderboard/放射医学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=放射医学主管技师) ，
Laboratory Technology (Junior)：[Leaderboard](leaderboard/检验技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=检验技术（士）)，
Laboratory Technology (Technician)：[Leaderboard](leaderboard/检验技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=检验技术（师）)，
Microbiology Testing Chief Technician Exam：[Leaderboard](leaderboard/微生物检验主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=微生物检验主管技师)，
Physical & Chemical Testing Chief Technician Exam：[Leaderboard](leaderboard/理化检验主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=理化检验主管技师)，
Clinical Laboratory Medicine Chief Technician Exam：[Leaderboard](leaderboard/临床医学检验主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学检验主管技师)， 
Pathology Attending Physician Exam：[Leaderboard](leaderboard/病理科主治医师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=病理科主治医师)，
Pathology Chief Technician Exam：[Leaderboard](leaderboard/病理学主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=病理学主管技师)，
Pathology Technology：[Leaderboard](leaderboard/主管技师-病理学技术.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=主管技师-病理学技术)， 
Rehabilitation Therapy Technology (Junior)：[Leaderboard](leaderboard/康复医学治疗技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学治疗技术（士）)，
Rehabilitation Therapy Technology (Technician)：[Leaderboard](leaderboard/康复医学治疗技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学治疗技术（师）)，
Rehabilitation Medicine & Therapy Chief Technician Exam：[Leaderboard](leaderboard/康复医学与治疗主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=康复医学与治疗主管技师)，
Oncology Technology (Junior)：[Leaderboard](leaderboard/肿瘤学技术（士）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤学技术（士）)，
Oncology Technology (Technician)：[Leaderboard](leaderboard/肿瘤学技术（师）.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤学技术（师）)，
Radiation Oncology Chief Technician Exam：[Leaderboard](leaderboard/肿瘤放射治疗主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=肿瘤放射治疗主管技师)，
Blood Transfusion Technology Chief Technician Exam：[Leaderboard](leaderboard/输血技术主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=输血技术主管技师)，
Disinfection Technology Chief Technician Exam：[Leaderboard](leaderboard/消毒技术主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=消毒技术主管技师)，
Medical Records Chief Technician Exam：[Leaderboard](leaderboard/病案信息主管技师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=病案信息主管技师)
<br><br>


### 3.5 Basic Medical Knowledge
（1）基础医学，[Leaderboard](leaderboard/基础医学.md)<br>
Medical Three Basics (San Ji)：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学三基)，
Medical Psychology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学心理学)，
Biochemistry & Molecular Biology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=生物化学与分子生物学)，
Cell Biology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-细胞生物学)，
Medical Immunology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学免疫学)，
Immunology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-免疫学)，
Pathophysiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-病理生理学)，  
Pathology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-病理学)，
Medical Genetics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学遗传学)，
Parasitology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-寄生虫学)，
Human Parasitology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-人体寄生虫学)，
Systematic Anatomy：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-系统解剖学)，
Anatomy：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-解剖学)，
Regional Anatomy：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-局部解剖学)，
Bioinformatics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-生物信息学)，
Physiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-生理学)，
Pharmacology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-药理学)，
Pharmaceutical Analysis：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-药物分析学)，
Medical Microbiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学微生物学)，
Histology & Embryology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-组织学与胚胎学)，
Medical Statistics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学统计学)
<br>

（2）Clinical Medicine，[Leaderboard](leaderboard/临床医学.md)<br>
Clinical Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学综合)，
Medical Imaging：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-医学影像学)，
Radiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-放射学)，
Laboratory Diagnostics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-实验诊断学)，
Neurology (Subject)：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-神经病学)，
Surgery：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-外科学)，
Dermatology & Venereology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-皮肤性病学)，
Pediatrics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-儿科学)，
Nuclear Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-核医学)，
Physical Diagnostics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-物理诊断学)，
Endodontics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-牙体牙髓病学)，
Fundamentals of Nursing：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-护理学基础)，
Nursing：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-护理学基础)，
Fundamentals of Nursing：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-基础护理学)，
Diagnostics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-诊断学)，
Ultrasound Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-超声医学)，
Dental Nursing：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-口腔护理学)，
Evidence-Based Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-循证医学)，
Epidemiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-流行病学)，
Oral Histopathology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-口腔组织病理学)，
Infectious Diseases：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-传染病学)，
Oral Anatomy & Physiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-口腔解剖生理学)，
Anesthesiology (Subject)：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-麻醉学)，
Interventional Radiology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=临床医学-介入放射学)
<br>

（3）Preventive Medicine与公共Hygiene，[Leaderboard](leaderboard/预防医学与公共卫生学.md)<br>
Preventive Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=预防医学)，
Hygiene：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=卫生学)，
Medical Ethics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学伦理学)
<br>

（4）Traditional Chinese Medicine与中药学，[Leaderboard](leaderboard/中医学与中药学.md)<br>
TCM Ophthalmology：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医眼科学)，
Synopsis of Golden Chamber Lecture Notes：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金匮要略讲义)，
Fundamentals of TCM Theory：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医基础理论)，
TCM Diagnostics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医诊断学)，
Traditional Chinese Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医学)，
Warm Disease Theory (Wenbingxue)：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=温病学)，
History of Chinese Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中国医学史)，
TCM Internal Medicine：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医内科学)，
TCM Pediatrics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中医儿科学)，
Treatise on Cold Damage (Shanghan Lun)：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=伤寒论)，
Huangdi Neijing Lecture Notes：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=内经讲义)
<br><br>


### 3.6 Medical Postgraduate Entrance Exam
Medical Postgraduate Entrance Exam，包含外科Nursing、Fundamentals of Nursing、西医综合等5个方向，参考[CMB](https://github.com/FreedomIntelligence/CMB)。☛☛See full leaderboard: [Medical Postgraduate Entrance Exam](leaderboard/医学考研.md)。<br>
(1) Surgical Nursing：[Leaderboard](leaderboard/医学考研-外科护理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学考研-外科护理学)，
(2) Fundamentals of Nursing：[Leaderboard](leaderboard/医学考研-基础护理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学考研-基础护理学)，
(3) Postgrad Entrance Exam Politics：[Leaderboard](leaderboard/考研政治.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=考研政治)，
(4) Comprehensive Western Medicine：[Leaderboard](leaderboard/医学考研-西医综合.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学考研-西医综合)，
(5) Comprehensive TCM：[Leaderboard](leaderboard/医学考研-中医综合.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=医学考研-中医综合)
<br><br>


### 3.7 Mental Health
目前包含4个子项：心理综合，Psychotherapy Attending Physician Exam，Psychological Counsellor Exam，Medical Psychology。☛☛See full leaderboard: [Mental Health](leaderboard/心理健康.md)。<br>
(1) Comprehensive Psychology：[Leaderboard](leaderboard/心理综合.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理综合)，
(2) Psychotherapy Attending Physician Exam：[Leaderboard](leaderboard/心理治疗学主治医师考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理治疗学主治医师考试)，
(3) Psychological Counsellor Exam：[Leaderboard](leaderboard/心理咨询师考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=心理咨询师考试)，
(4) Medical Psychology：[Leaderboard](leaderboard/基础医学-医学心理学.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基础医学-医学心理学)
<br><br><br>



## 4、FinanceLeaderboard
☛☛See full leaderboard: [Finance](leaderboard/金融.md)<br>

### 4.1 Finance & Accounting
☛☛See full leaderboard: [Finance & Accounting](leaderboard/财务.md)。<br>
Junior Accounting Title：[Leaderboard](leaderboard/初级会计职称.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级会计职称)，
Certified Public Accountant (CPA)：[Leaderboard](leaderboard/注册会计师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=注册会计师)，
Accounting Practice Qualification：[Leaderboard](leaderboard/会计从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=会计从业资格)，
Auditor Exam：[Leaderboard](leaderboard/审计师考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=审计师考试)，
Certified Tax Agent：[Leaderboard](leaderboard/注册税务师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=注册税务师)，
Certified Management Accountant：[Leaderboard](leaderboard/注册管理会计师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=注册管理会计师)

### 4.2 Banking
☛☛See full leaderboard: [Banking](leaderboard/银行.md)。<br>
Banking Junior Qualification：[Leaderboard](leaderboard/银行初级资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=银行初级资格)，
Banking Intermediate Qualification：[Leaderboard](leaderboard/银从中级资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=银从中级资格)，
Banking Practice Qualification：[Leaderboard](leaderboard/银行从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=银行从业资格)

### 4.3 Insurance
☛☛See full leaderboard: [Insurance](leaderboard/保险.md)。<br>
Insurance Practice Qualification：[Leaderboard](leaderboard/保险从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险从业资格)

### 4.4 Securities
☛☛See full leaderboard: [Securities](leaderboard/证券.md)。<br>
Securities Special Qualification Exam：[Leaderboard](leaderboard/证券专项考试.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=证券专项考试)，
Securities Practice Qualification：[Leaderboard](leaderboard/证券从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=证券从业资格)

### 4.5 Other Financial Qualification Exams
☛☛See full leaderboard: [Other Financial Qualification Exams](leaderboard/其他金融资格考试.md)。<br>
Junior Economist Exam：[Leaderboard](leaderboard/初级经济师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=初级经济师)，
Intermediate Economist Exam：[Leaderboard](leaderboard/中级经济师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中级经济师)，
Anti-Counterfeit Currency Knowledge：[Leaderboard](leaderboard/反假货币知识.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=反假货币知识)，
Futures Practice Qualification：[Leaderboard](leaderboard/期货从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=期货从业资格)，
AFP Financial Planner：[Leaderboard](leaderboard/金融理财师AFP.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融理财师AFP)，
Fund Practice Qualification：[Leaderboard](leaderboard/基金从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=基金从业资格)，
Gold Trading Practice Qualification：[Leaderboard](leaderboard/黄金从业资格.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=黄金从业资格)，
China Actuary Exam：[Leaderboard](leaderboard/中国精算师.md)|[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中国精算师)

### 4.6 Basic Financial Knowledge
☛☛See full leaderboard: [Basic Financial Knowledge](leaderboard/金融基础知识.md)。<br>
Finance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融学)，
Corporate Strategy & Risk Management：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公司战略与风险管理)，
Macroeconomics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=宏观经济学)，
Financial Markets：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融市场学)，
Accounting：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=会计学)，
Cost Accounting：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=成本会计学)，
Money & Banking：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=货币金融学)，
Political Economy：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=政治经济学)，
Investment Studies：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=投资学)，
Econometrics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=计量经济学)，
Corporate Finance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=公司金融学)，
Public Finance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=财政学)，
Commercial Bank Finance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=商业银行金融学)，
Managerial Accounting：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=管理会计学)，
Central Banking：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中央银行学)，
Auditing：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=审计学)，
International Economics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=国际经济学)，
Intermediate Financial Accounting：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中级财务会计)，
Financial Management：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=财务管理学)，
Microeconomics：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=微观经济学)，
International Finance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=国际金融学)，
Financial Engineering：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融工程学)，
Economic Law：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=经济法)，
Advanced Financial Accounting：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=高级财务会计)，
Insurance Studies：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险学)

### 4.7 Financial Applications
☛☛See full leaderboard: [Financial Applications](leaderboard/金融应用.md)。<br>
Insurance Knowledge Interpretation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险知识解读)，
Financial Terminology Explanation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融术语解释)，
Licensed Physician Qualification Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融知识-执业医师资格考试)，
Wealth Management Knowledge Interpretation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=理财知识解读)，
Licensed Pharmacist Qualification Exam：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融知识-执业药师资格考试)，
Financial Document Extraction：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融文档抽取)，
Analytical Opinion Extraction：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融认知-研判观点提取)，
Financial Sentiment Recognition：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融情绪识别)，
Insurance Slot Filling：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险槽位识别)，
Insurance Intent Understanding：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险意图理解)，
Financial Intent Understanding：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融意图理解)，
Insurance Attribute Extraction：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险属性抽取)，
Insurance Clause Interpretation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=保险条款解读)，
Financial Product Analysis：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融产品分析)，
Financial Numerical Computation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融数值计算)，
Financial Event Interpretation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融事件解读)，
内容生成-Investor Education Script Generation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融投教话术生成)，
内容生成-Text Summarization：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融文本总结归纳)，
内容生成-Marketing Copy Generation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融营销文案生成)，
内容生成-News Headline Generation：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融资讯标题生成)，
安全合规-Financial Compliance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融合规性)，
安全合规-Financial Issue Recognition：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融问题识别)，
安全合规-Information Security Compliance：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融信息安全合规)，
安全合规-Financial Factuality：[badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=金融事实性)
<br><br><br>


## 5、Law & Public AdministrationLeaderboard
☛☛See full leaderboard: [Law & Public Administration](leaderboard/法律与行政公务.md)<br>

### 5.1 Bar Exam (Legal Qualification)
#### （1）JEC-QA-KD
选择题，共1000道，参考[AGIEval](https://github.com/ruixiangcui/AGIEval)。<br>
See full leaderboard: [JEC-QA-KD](leaderboard/JEC-QA-KD.md)，☛查看[JEC-QA-KD：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=JEC-QA-KD)
<br>

#### （2）JEC-QA-CA
选择题，共1000道，参考[AGIEval](https://github.com/ruixiangcui/AGIEval)。<br>
See full leaderboard: [JEC-QA-CA](leaderboard/JEC-QA-CA.md)，☛查看[JEC-QA-CA：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=JEC-QA-CA)
<br>

#### （3）法律综合
See full leaderboard: [法律综合](leaderboard/法律综合.md)，☛查看[法律综合：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=法律综合)
<br><br><br>


### 5.2 Civil Service Exam
Civil Service Exam行测选择题，共651道，参考[AGIEval](https://github.com/ruixiangcui/AGIEval)。
Example evaluation sample：
> 某乡镇进行新区规划，决定以市民公园为中心，在东南西北分别建设一个特色社区。这四个社区分别定为，文化区、休闲区、商业区和行政服务区。已知行政服务区在文化区的西南方向，文化区在休闲区的东南方向。   
根据以上陈述，可以得出以下哪项？   
(A)市民公园在行政服务区的北面 (B)休闲区在文化区的西南 (C)文化区在商业区的东北 (D)商业区在休闲区的东南   
>  

See full leaderboard: [Civil Service Exam](leaderboard/考公.md)<br>
☛查看[Civil Service Exam：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=kaogong-so)
<br><br><br>



## 6、Reasoning & Mathematics Leaderboard
☛☛See full leaderboard: [推理与Mathematics计算](leaderboard/推理与数学计算.md)<br>

### 6.1 Deductive Reasoning
Deductive Reasoning（modus_tollens）选择题，共123道，参考[ISP](https://arxiv.org/abs/2306.09479)。

Example evaluation sample：
> 考虑以下语句：  
1.如果约翰是个好父母，那么约翰就是严格但公平的。2.约翰不严格但公平。 结论：因此，约翰不是一个好父母。
问题：根据陈述1.和2.，结论是否正确？   
回答： (A) 否   (B) 是   
>  

See full leaderboard: [Deductive Reasoning](leaderboard/演绎推理.md)<br>
☛查看[Deductive Reasoning：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=演绎推理)
<br><br>


### 6.2 Commonsense Reasoning
Commonsense Reasoning选择题，共99道，参考[ISP](https://arxiv.org/abs/2306.09479)。

Example evaluation sample：
> 以下是关于常识的选择题。   
问题：当某人把土豆放到篝火边的余烬中，此时余烬并没有在   
A、释放热量  B、吸收热量   
>      

See full leaderboard: [Commonsense Reasoning](leaderboard/常识推理.md)<br>
☛查看[Commonsense Reasoning：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=常识推理)
<br><br>


### 6.3 Symbolic Reasoning (BBH)
学术界最常用的符号推理评测集，包含23个子任务，详细介绍See[BBH](https://nonelinear.com/static/benchmarks.html)。
Example evaluation sample：
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

See full leaderboard: [BBH](leaderboard/bbh.md)<br>
☛查看[BBH符号推理：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=BBH)
<br><br>


### 6.4 Arithmetic Ability
考查大模型的Mathematics基础能力之算数能力，测试题目为1000以内的整数加减法、不超过2位有效数字的浮点数加减乘除。
example：166 + 215 + 53 = ？，0.97 + 0.4 / 4.51 = ？

See full leaderboard: [Arithmetic Ability](leaderboard/算术能力.md)<br>
☛查看[Arithmetic Ability：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=算术能力)
<br><br>


### 6.5 Table Q&A
专门考查大模型对表格的理解分析能力，常用于数据分析。    
Example evaluation sample：
> 姓名,年龄,性别,国籍,身高(cm),体重(kg),学历   
张三,28,男,中国,180,70,本科   
Lisa,33,女,美国,165,58,硕士   
Paulo,41,男,巴西,175,80,博士   
Miyuki,25,女,日本,160,50,大专   
Ahmed,30,男,埃及,175,68,本科   
Maria,29,女,墨西哥,170,65,硕士   
Antonio,36,男,西班牙,182,75,博士  
基于这个表格回答：学历最低的是哪国人？
> 

See full leaderboard: [Table Q&A](leaderboard/表格问答.md)<br>
☛查看[Table Q&A：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=表格问答)
<br><br>


### 6.6 Table Summarization
专门考查大模型对表格的分析总结能力，常用于数据分析、文章撰写，没有固定的标准答案，但容易相对客观地分辨好坏。
Example evaluation sample（由于例子过长，部分数据予以省略）：
> |类别|机构|大模型|准确率|平均耗时|平均消耗token|花费/千次（元）|排名（准确率）|  
> |---|---|-----|-------------------|-------|-----------|-----------|-----------|  
> |商用|豆包|doubao-seed-1-6-thinking-250715|87.5|37s|1976|14.6|1|   
> |商用|百度|ERNIE-4.5-Turbo-32K|84.7|33s|676|1.8|2|   
> |商用|腾讯|hunyuan-t1-20250711|84.7|37s|2465|9.2|3|   
> |商用|腾讯|hunyuan-turbos-20250716|83.9|24s|1288|2.3|4|   
> |……|……|……|……|……|……|……|……|   
> -------------------------   
> 已知新模型为：GLM-4.5,GLM-4.5-Air,GLM-4.5-Flash,step-3。   
> 基于以上表格写一段总结，格式为：“xx机构、xx机构……占据前5（机构名不要重复），然后描述开源模型和商用模型的分布。新模型中，xx排第xx，xx排第xx……（排名由高到低）”。严格按照表格中的模型名称、机构名称。   
>   

See full leaderboard: [Table Summarization](leaderboard/表格总结.md)<br>
☛查看[Table Summarization：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=表格总结)
<br><br>


### 6.7 High School Olympiad Mathematics
2024预赛试题，参考[Math24o](https://github.com/CLUEbenchmark/Math24o)。
Example evaluation sample：
> 设集合 $S=\{1, 2, 3, \cdots, 9 9 7, 9 9 8 \}$，集合 $S$ 的 $k$ 个 $499$ 元子集 $A_{1},A_{2}, \cdots, A_{k}$ 满足：对 $S$ 中任一二元子集 $B$，均存在 $i \in\{1, 2, \cdots, k \}$，使得 $B \subset A_{i}$。求 $k$ 的最小值。
> 

See full leaderboard: [高中奥林匹克Mathematics竞赛](leaderboard/高中奥数.md)<br>
☛查看[高中奥林匹克Mathematics竞赛：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=高中奥数)
<br><br>


### 6.8 Middle School Olympiad Mathematics TODO
<br>


### 6.9 Primary School Olympiad Mathematics
See full leaderboard: [Primary School Olympiad Mathematics](leaderboard/小学奥数.md)<br>
☛查看[Primary School Olympiad Mathematics：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=小学奥数一年级)
<br><br>


### 6.10 Map Reasoning TODO
### 6.11 Spatial Reasoning TODO
<br>


### 6.12 Sudoku
See full leaderboard: [Sudoku](leaderboard/数独.md)<br>
☛查看[Sudoku：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=数独入门)
<br>


### 6.13 Currency Amount Numeral Conversion TODO
### 6.14 Date Calculation TODO
<br><br><br>



## 7、Language & Instruction FollowingLeaderboard
☛☛See full leaderboard: [Language & Instruction Following](leaderboard/语言与指令遵从.md)<br>

### 7.1 Idiom Comprehension
给定上下文，选择最匹配的成语。

Example evaluation sample：
> 说完作品的优点,咱们再来聊聊为何说它最后的结局____,片子本身提出的话题观点很尖锐,“扶弟魔”也成为众多当代年轻人婚姻里的不定因素,所以对于这种过于敏感的东西,片子的结局仅仅只是以弟弟的可爱化解了姐姐的心结,最后选择陪伴照顾...   
给上文空格处选择最合适的成语或俗语：   
(A) 有条有理   (B) 偏听偏信   (C) 狗尾续貂   (D) 半壁江山   (E) 身家性命   (F) 胆小如鼠   (G) 独善其身    
> 

See full leaderboard: [Idiom Comprehension](leaderboard/成语理解.md)<br>
☛查看[Idiom Comprehension：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=成语理解)
<br><br>


### 7.2 Sentiment Analysis
分析用户评论的情感属性，消极或积极。

Example evaluation sample：
> 用了几天，发现很多问题，无线网容易掉线，屏幕容易刮花，打开网页容易死掉，不值的买   
以上用户评论是正面还是负面？    
(A) 负面   (B) 正面   
>    

See full leaderboard: [Sentiment Analysis](leaderboard/情感分析.md)<br>
☛查看[Sentiment Analysis：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=情感分析)
<br><br>


### 7.3 Textual Entailment
Textual Entailment，判断两个句子之间的语义关系：蕴含、中立、矛盾，参考[OCNLI](https://arxiv.org/abs/2010.05444)。

Example evaluation sample：
> 句子一：农机具购置补贴覆盖到全国所有农牧业县(场),中央财政拟安排资金130亿元,比上年增加90亿元   
句子二：按农民人数发放补贴  
以上两个句子是什么关系？   
(A)蕴含  (B)中立  (C)矛盾   
>   

See full leaderboard: [Textual Entailment](leaderboard/文本蕴含.md)<br>
☛查看[Textual Entailment：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=文本蕴含)
<br><br>


### 7.4 Text Classification
Example evaluation sample：
> 将下列单词按词性分类。    
> 狗，追，跑，大人，高兴，树

See full leaderboard: [Text Classification](leaderboard/文本分类.md)<br>
☛查看[Text Classification：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=文本分类)
<br><br>


### 7.5 Information Extraction
Example evaluation sample：  
> “中信Banking3亿元，交通Banking增长约2.7亿元，光大Banking约1亿元。”    
> 提取出以上文本中的所有组织机构名称

See full leaderboard: [Information Extraction](leaderboard/信息抽取.md)<br>
☛查看[Information Extraction：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=信息抽取)
<br><br>


### 7.6 Reading Comprehension
Reading Comprehension能力是一种符合能力，考查针对给定信息的理解能力。
依据给定信息的种类，可以细分为：文章问答、Table Q&A、对话问答……    
Example evaluation sample：
> 牙医：好的，让我们看看你的牙齿。从你的描述和我们的检查结果来看，你可能有一些牙齦疾病，导致牙齿的神经受到刺激，引起了敏感。此外，这些黑色斑点可能是蛀牙。  
病人：哦，真的吗？那我该怎么办？   
牙医：别担心，我们可以为你制定一个治疗计划。我们需要首先治疗牙龈疾病，然后清除蛀牙并填充牙洞。在此过程中，我们将确保您感到舒适，并使用先进的技术和材料来实现最佳效果。   
病人：好的，谢谢您，医生。那么我什么时候可以开始治疗？   
牙医：让我们为您安排一个约会。您的治疗将在两天后开始。在此期间，请继续刷牙，使用牙线，并避免吃过于甜腻和酸性的食物和饮料。   
病人：好的，我会的。再次感谢您，医生。   
牙医：不用谢，我们会尽最大的努力帮助您恢复健康的牙齿。   
基于以上对话回答：病人在检查中发现的牙齿问题有哪些？
> 

See full leaderboard: [Reading Comprehension](leaderboard/阅读理解.md)<br>
☛查看[Reading Comprehension：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=阅读理解)
<br><br>


### 7.7 Pronoun Resolution
中文指代消解任务，参考[CLUEWSC2020](https://github.com/CLUEbenchmark/CLUEWSC2020)。
Example evaluation sample：
> 少平仍然不知道怎样给奶奶说清他姐夫的事，就只好随口说：“他犯了点错误，人家让他劳教！”  
上述文本中的“他犯了点错误”中的“他”是指少平吗？   选项：(A)是   (B)否      
>    

See full leaderboard: [Pronoun Resolution](leaderboard/代词理解.md)<br>
☛查看[Pronoun Resolution：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=代词理解)
<br><br>


### 7.8 Classical Poetry Matching
中国古典诗歌匹配，给定中国古典诗歌的现代问描述，要求从候选的四句诗中选出与现代文描述语义匹配的那一句。
利用古典诗歌和现代文翻译的平行语料构建正确选项，并利用正确选项从古代诗歌语料库中利用相似检索构造出错误候选。
参考[CCPM](https://github.com/THUNLP-AIPoet/CCPM)。
Example evaluation sample：
> 昏暗的灯熄灭了又被重新点亮。   
上述文本最匹配下面哪句诗：   
(A)渔灯灭复明   (B)残灯灭又然   (C)残灯暗复明   (D)残灯灭又明   
>    

See full leaderboard: [Classical Poetry Matching](leaderboard/诗词匹配.md)<br>
☛查看[Classical Poetry Matching：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=诗词匹配)
<br><br>


### 7.9 Chinese Instruction Following
参考谷歌IFEval，并将其翻译和适配到中文，精选9类25种指令，说明如下：
![lin](pic/IFEval.jpg)

See full leaderboard: [IFEval](leaderboard/中文指令遵从.md)<br>
☛查看[Chinese Instruction Following：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=中文指令遵从)
<br><br>


### 7.10 Chinese Character Glyphs
See full leaderboard: [Chinese Character Glyphs](leaderboard/汉字字形.md)<br>
☛查看[Chinese Character Glyphs：badcase](https://nonelinear.com/static/badcase/badcase-of-benchmark.html?benchmark=汉字字形)
<br><br>


### 7.11 Hanyu Pinyin TODO
### 7.12 Typo Detection TODO
### 7.13 Sentence Comprehension TODO
### 7.14 Punctuation TODO
### 7.15 Traditional/Simplified Chinese Conversion TODO
### 7.16 Language Identification TODO
<br><br><br>


## 8、Agent & Tool UseLeaderboard
计算TAU和BFCL-V3的平均分。<br>
☛☛See full leaderboard: [Agent & Tool UseLeaderboard](leaderboard/agent与工具调用.md)<br>

### 8.1 TAU
See full leaderboard: [TAU](leaderboard/TAU.md)<br>
#### (1) TAU-airline
See full leaderboard: [TAU-airline](leaderboard/TAU-airline.md)<br>

#### (2) TAU-retail
See full leaderboard: [TAU-retail](leaderboard/TAU-retail.md)
<br><br>


### 8.2 BFCL-V3
BFCL-V3是加州大学伯克利分校发布的工具调用评测集，首创多轮、多步函数调用场景，并通过API状态验证评估模型真实交互能力，是目前最权威的大模型工具使用基准之一。
<br>See full leaderboard: [BFCL-V3](leaderboard/BFCL-V3.md)
<br><br><br>



## 9、CodingLeaderboard
评估大模型编程能力。See full leaderboard: [Coding](leaderboard/coding.md)<br>

### 9.1 livecodebench
[LiveCodeBench](https://github.com/LiveCodeBench/LiveCodeBench) 为大型语言模型（LLM）的编程能力提供全面且无数据污染的评估。具体而言，LiveCodeBench 持续从三大竞赛平台——LeetCode、AtCoder 和 CodeForces——的赛事中随时间推移不断收集新题目。
<br>See full leaderboard: [livecodebench](leaderboard/livecodebench.md)
<br><br>


### 9.2 Terminal-Bench-2.0
[Terminal-Bench](https://github.com/harbor-framework/terminal-bench-2)是一个热门基准测试，用于评估智能体和语言模型在容器化环境中执行有价值工作的能力。测试任务包括蛋白质合成组装、异步代码调试以及安全漏洞修复等。
<br>See full leaderboard: [Terminal-Bench-2.0](leaderboard/Terminal-Bench-2.0.md)
<br><br><br>


## 10、Integrating LMArena and AA Scores
整合我们ReLE评测（中文）和LMArena（英文）、Artificial Analysis（简称AA，英文）Leaderboard数据。

| 大模型                                    | ReLE评测（中文）   |    | AA-Intelligence（英文）   | AA-Coding（英文）   | AA-Math（英文）   |    | LMArena-Text-overall（英文）   | LMArena-Text-Coding（英文）   | LMArena-WebDev（英文）   |
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

完整分数See[LMArena+AA](LMArena+AA.md)
<br><br>


## 🌐Scores by Ability
评分方法：从各个维度给大模型打分，每个维度都对应一个评测数据集，包含若干道题。
每道题依据大模型回复质量给1~5分，将评测集内所有题的得分累加并归一化为100分制，即作为最终得分。

所有评分数据See [alldata](leaderboard/alldata.md)
<br><br>


## Why build this leaderboard?
- 大模型百花齐放，也参差不齐。不少媒体的宣传往往夸大其词，避重就轻，容易混淆视听；而某些公司为了PR，也过分标榜自己大模型的能力，动不动就“达到chatgpt水平”，动不动就“Domestic (China)第一”。
所谓“外行看热闹，内行看门道”，业界急需一股气流，摒弃浮躁，静下心来打磨前沿技术，真真正正用技术实力说话。这就少不了一个公开、公正、公平的大模型评测系统，把各类大模型的优点、不足一一展示出来。
如此，大家既能把握当下的发展水平、与Overseas顶尖技术的差距，也能更加清晰地看明白未来的努力方向，而不被资本热潮、舆论热潮所裹挟。
- 对于产业界来说，特别是对于不具备大模型研发能力的公司，熟悉大模型的技术边界、高效有针对性地做大模型技术选型，在现如今显得尤为重要。
而一个公开、公正、公平的大模型评测系统，恰好能够提供应有的助力，避免重复造轮子，避免因技术栈不同而导致不必要的争论，避免“鸡同鸭讲”。
- 对于大模型研发人员，包括对大模型技术感兴趣的人、学术界看中实践的人，各类大模型的效果对比，反应出了背后不同技术路线、技术方法的有效性，这就提供了非常好的参考意义。
不同大模型的相互参考、借鉴，帮忙大家躲过不必要的坑、避免重复实验带来的资源浪费，有助于整个大模型生态圈的良性高效发展。
<br><br>


## 联系我们（非线智能 ReLE benchmark团队）
### 大模型评测交流群
先加小编微信，后拉入群，备注“来源github，加群”<br>
![lin](pic/qrcode-wxgroup.jpg)
<br><br><br>
### 大模型评测微信公众号
关注大模型评测微信公众号，及时Get it at 最新评测信息<br>
![lin](pic/qrcode-gzh.jpg)
<br><br><br>

---

## 📖如何引用 ReLE 评测（Cite Us）

若您在自己的论文、报告或开源项目中使用了 ReLE（ chinese-llm-benchmark ）数据、结果或代码，请按以下格式引用，帮助我们持续维护开源评测生态。

### 中文引用（GB/T 7714）
ReLE 评测组. ReLE：中文 AI 大模型能力评测数据集与开放Leaderboard[EB/OL]. GitHub, 2023-06-04[2025-12-06]. https://github.com/jeinlee1991/chinese-llm-benchmark. DOI: 10.5281/zenodo.xxxxxxx.

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

###  release号说明
ReLE 采用语义化 release号（`主版本.次版本.修订号`）。  
- 主 release：重大框架或指标权级调整  
- 次 release：新增领域、子榜单或>10% 题库扩充  
- 修订号：bug 修复、样本去噪、模型增补  

请在引用时注明您使用的 **精确 tag**（如 `v5.8.5`），以保证结果可复现。

