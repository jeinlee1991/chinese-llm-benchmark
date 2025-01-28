
# CLiB中文大模型能力评测榜单（持续更新）
- 目前已囊括164个大模型，覆盖chatgpt、gpt-4o、谷歌gemini、Claude3.5、百度文心一言、千问、百川、讯飞星火、商汤senseChat、minimax等商用模型，
以及deepseek-v3、qwen2.5、llama3.3、phi-4、glm4、书生internLM2.5等开源大模型。
- 模型来源涉及国内外大厂、大模型创业公司、高校研究机构。
- 支持多维度能力评测，包括分类能力、信息抽取、阅读理解、数据分析、指令遵从、算术运算、初中数学、符号推理BBH、代词理解CLUEWSC、诗词匹配CCPM、公务员考试、律师资格考试、中文编码效率。
- 不仅提供能力评分排行榜，也提供所有模型的原始输出结果！有兴趣的朋友可以自己打分、自己排行！

## 目录
- [🔄最近更新](#最近更新)
- [⚓TODO](#todo)
- [📝大模型基本信息](#大模型基本信息)
- [📊排行榜](#-排行榜)
  - [综合能力排行榜](#1、综合能力排行榜)
    - 商用大模型排行榜（含开源模型的付费API）
      - 输出价格30元及以上
      - 输出价格5~30元
      - 输出价格1~5元
      - 输出价格1元以下
    - 开源大模型排行榜
      - 5B以下
      - 5B~20B
      - 20B以上
  - [【学科知识】高考排行榜](#2、【学科知识】高考排行榜)
  - [【推理】常识推理排行榜](#3、【推理】常识推理排行榜)
  - [【推理】公务员考试排行榜](#4、【推理】公务员考试排行榜)
  - [【推理】律师资格考试排行榜](#5、【推理】律师资格考试排行榜)
  - [【推理】符号推理BBH排行榜](#6、【推理】符号推理BBH排行榜)
  - [【数学计算】初中数学排行榜](#7、【数学计算】初中数学排行榜)
  - [【数学计算】算术能力排行榜](#8、【数学计算】算术能力排行榜)
  - [【语言理解】文本蕴含排行榜](#9、【语言理解】文本蕴含排行榜)
  - [【语言理解】分类能力排行榜](#10、【语言理解】分类能力排行榜)
  - [【语言理解】信息抽取能力排行榜](#11、【语言理解】信息抽取能力排行榜)
  - [【语言理解】阅读理解能力排行榜](#12、【语言理解】阅读理解能力排行榜)
  - [【语言理解】代词理解CLUEWSC排行榜](#13、【语言理解】代词理解CLUEWSC排行榜)
  - [【传统文化】诗词匹配CCPM排行榜](#14、【传统文化】诗词匹配CCPM排行榜)
  - [数据分析排行榜](#15、数据分析排行榜)
  - [中文指令遵从排行榜](#16、中文指令遵从排行榜)
  - [中文编码效率排行榜](#17、中文编码效率排行榜)
- [🌐各项能力评分](#🌐各项能力评分)
- [⚖️原始评测数据](#⚖️原始评测数据)
- [为什么做榜单？](#为什么做榜单)


## 最近更新
- [2025/1/29] 发布v2.13版本评测榜单
  - 新增常识推理排行榜、文本蕴含（语言理解）排行榜，并计入总分
  - 阅读理解评测样本增加至600多个，并更新各模型评分
- [2025/1/25] 发布v2.12版本评测榜单
  - 新增高考榜单及各学科细分榜单（生物、化学、语文、地理、历史、数学、物理），并以各科平均分（100分制）计入总分
- [2025/1/23] 发布v2.11版本评测榜单
  - 公务员考试kaogong、律师资格考试JEC-QA开始计入总分
  - 新增4个模型：mistral-small、Hermes-3-Llama-3.1-405B、mistral-large、360gpt2-o1，☛查看[模型完整信息](https://easyllm.site/static/models.html)
- [2025/1/22] 发布v2.10版本评测榜单
  - 新增律师资格考试JEC-QA榜单，暂不计入总分
  - 新增7个模型：ministral-3b、Mistral-7B-Instruct-v0.3、Mistral-Nemo-Instruct-2407、ministral-8b、Mixtral-8x7B-Instruct-v0.1、Llama-3.1-Nemotron-70B-Instruct-fp8、WizardLM-2-8x22B，☛查看[模型完整信息](https://easyllm.site/static/models.html)
- [2025/1/20] 发布v2.9版本评测榜单
  - 新增公务员考试kaogong榜单，暂不计入总分
  - 新增5个模型：Llama-3.2-1B-Instruct、Llama-3.2-3B-Instruct、Llama-3.1-8B-Instruct-fp8、Llama-3.3-70B-Instruct-fp8、Llama-3.1-70B-Instruct-fp8，☛查看[模型完整信息](https://easyllm.site/static/models.html)
- [2025/1/17] 发布v2.8版本评测榜单
  - 新增9个模型：gemini-2.0-flash-exp、phi-4、gemini-1.5-flash-8b、360gpt-turbo、step-1-flash、Llama-3.3-70B-Instruct、360gpt-pro、360gpt2-pro、step-1-8k，☛查看[模型完整信息](https://easyllm.site/static/models.html)
  - 新增o1-mini、o1-preview的初中数学成绩
  - 删除陈旧的模型：abab5.5-chat、abab5.5s-chat
- [2025/1/7] 发布v2.7版本评测榜单
  - 新增代词理解CLUEWSC榜单（比如“他”是指谁）、诗词匹配CCPM榜单
  - 新增5个模型：Claude-3.5-Sonnet、gemma-2-27b-it、Llama-3.1-405B-Instruct、Baichuan4-Air、Baichuan4-Turbo，☛查看[模型完整信息](https://easyllm.site/static/models.html)
  - 删除陈旧的模型：Baichuan3-Turbo、qwen2-72b-instruct、Qwen2-7B-Instruct、qwen2-1.5b-instruct、qwen2-0.5b-instruct、qwen2-57b-a14b-instruct
- [2024/12/28] 发布v2.6版本评测榜单
  - 新增BBH（学术界常用符号推理评测集）榜单，并计入总分
  - 将初中数学（七/八/九年级）成绩计入总分
  - 删除陈旧的模型：deepseek-chat-v2、Llama-3-70B-Instruct、Llama-3-8B-Instruct、MiniCPM-2B-dpo、minimax-abab6.5-chat、DeepSeek-V2-Lite-Chat、internlm2-chat-1_8b
- [2024/12/27] 发布v2.5版本评测榜单
  - 新增Grade8Math-zh（八年级数学）、Grade9Math-zh（九年级数学）榜单
  - 新增6个模型：deepseek-chat-v3、abab7-chat-preview、hunyuan-standard、hunyuan-large、hunyuan-turbo、SenseChat-5，☛查看[模型完整信息](https://easyllm.site/static/models.html)
- [2024/12/25] 发布v2.4版本评测榜单
  - 新增Grade7Math-zh（七年级数学）榜单
  - 删除陈旧的模型：Phi-3-mini-128k-instruct、Qwen1.5系列、openbuddy-llama3-8b、yi-large、yi-large-turbo、yi-medium、yi-spark、internlm2-chat-20b、internlm2-chat-7b、gpt-4-turbo、gpt-3.5-turbo
- [2024/10/20] 发布v2.3版本评测榜单
  - 新增6个模型：yi-lightning、gemini-1.5-flash、gemini-1.0-pro、gemini-1.5-pro、GLM-4-Long、GLM-4-Plus
  - 更新4个模型：GLM4、qwen-max、ERNIE-4.0-Turbo-8K、ERNIE-3.5-8K
  - 删除陈旧的模型：Baichuan2-13B-Chat、Baichuan2-7B-Chat、deepseek-llm-67b-chat、gpt4、gemma-2b-it、gemma-7b-it
- [2024/9/29]v2.2版本，[2024/8/27]v2.1版本，[2024/8/7]v2.0版本，[2024/7/26]v1.21版本，[2024/7/15]v1.20版本，[2024/6/29]v1.19版本，[2024/6/2]v1.18版本，[2024/5/8]v1.17版本，[2024/4/13]v1.16版本，[2024/3/20]v1.15版本，[2024/2/28]v1.14版本，[2024/1/29]v1.13版本
- 2023年：[2023/12/10]v1.12版本，[2023/11/22]v1.11版本，[2023/11/5]v1.10版本，[2023/10/11]v1.9版本，[2023/9/13]v1.8版本，[2023/8/29]v1.7版本，[2023/8/13]v1.6版本，[2023/7/26]v1.5版本， [2023/7/18]v1.4版本， [2023/7/2]v1.3版本， [2023/6/17]v1.2版， [2023/6/10]v1.1版本， [2023/6/4]v1版本

各版本更新详情：[CHANGELOG](CHANGELOG.md)

## TODO
- 引入更多维度的评测：代码能力、开放域问答、多轮对话、头脑风暴、翻译……
- 评测维度更细分，比如信息抽取可以细分时间实体抽取能力、地址实体抽取能力……
- 海纳百川，整合各类评测榜单，扩充细分领域榜单（比如教育领域、医疗领域）
- 加入更多评测数据，使得评测得分越来越有说服力

## 大模型基本信息
价格单位：元/1M tokens，即元每百万token   

|model|	producer|open-source|price_input|price_output|直接体验|download|paper|badcase|
|---|---|---|---|---|---|---|---|---|
|GLM-4-Flash|	智谱AI|	No|	0.0|	0.0|[link](https://easyllm.site/static/modelcompare.html)|	/|	[link](https://arxiv.org/abs/2406.12793)|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=GLM-4-Flash)|
|ERNIE-Speed-8K|	百度|	No|	0.0|	0.0|[link](https://easyllm.site/static/modelcompare.html)|	/|	/|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=ERNIE-Speed-8K)|
|internlm2_5-7b-chat|	上海人工智能实验室|	Yes|	0.3|	0.3|[link](https://easyllm.site/static/modelcompare.html)|	[link](https://www.modelscope.cn/models/Shanghai_AI_Laboratory/internlm2_5-7b-chat)|	/|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=internlm2_5-7b-chat)|
|Yi-1.5-9B-Chat|	零一万物|	Yes|	0.4|	0.4|[link](https://easyllm.site/static/modelcompare.html)|	[link](https://www.modelscope.cn/models/01ai/Yi-1.5-9B-Chat/)|	[link](https://arxiv.org/abs/2403.04652)|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=Yi-1.5-9B-Chat)|
|Llama-3.1-8B-Instruct|	meta|	Yes|	0.4|	0.4|[link](https://easyllm.site/static/modelcompare.html)|	[link](https://modelscope.cn/models/llm-research/meta-llama-3.1-8b-instruct)|	[link](https://arxiv.org/abs/2407.21783)|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=Llama-3.1-8B-Instruct)|
|Doubao-lite-32k|	豆包|	No|	0.3|	0.6|[link](https://easyllm.site/static/modelcompare.html)|	/|	/|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=Doubao-lite-32k)|
|glm-4-9b-chat|	智谱AI|	Yes|	0.6|	0.6|[link](https://easyllm.site/static/modelcompare.html)|	[link](https://www.modelscope.cn/models/ZhipuAI/glm-4-9b-chat)|	[link](https://arxiv.org/abs/2406.12793)|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=glm-4-9b-chat)|
|gemma-2-9b-it|	google|	Yes|	0.6|	0.6|[link](https://easyllm.site/static/modelcompare.html)|	[link](https://www.modelscope.cn/models/LLM-Research/gemma-2-9b-it)|	[link](https://arxiv.org/abs/2408.00118)|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=gemma-2-9b-it)|
|qwen2.5-7b-instruct|	阿里巴巴|	Yes|	1.0|	2.0|[link](https://easyllm.site/static/modelcompare.html)|	[link](https://modelscope.cn/models/qwen/Qwen2.5-7B-Instruct)|	/|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=qwen2.5-7b-instruct)|
|gemini-1.5-flash|	google|	No|	0.5|	2.2|[link](https://easyllm.site/static/modelcompare.html)|	/|	/|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=gemini-1.5-flash)|
|gpt-4o-mini|	openAI|	No|	1.1|	4.3|[link](https://easyllm.site/static/modelcompare.html)|	/|	[link](https://arxiv.org/abs/2303.08774)|	[link](https://easyllm.site/static/badcase/badcase-of-llm.html?model=gpt-4o-mini)|
|...|...|...|...|...|...|...|...|...|

更多模型信息详见：
- [大模型资源汇总（商用及开源）](https://easyllm.site/static/models.html)
- [开源大模型发布历史](LLM-history.md)
<br><br>

## 📊 排行榜
### 1、综合能力排行榜
综合能力得分为分类能力、信息抽取、阅读理解、数据分析、指令遵从、算术运算、初中数学、符号推理BBH、代词理解CLUEWSC、诗词匹配CCPM、公务员考试kaogong、律师资格考试JEC-QA、高考、常识推理、文本蕴含等15项得分的平均值。
![lin](pic/total.png)    
详细数据见[total](leaderboard/total.md)
<br>

#### 1.1、商用大模型排行榜（含开源模型的付费API）
##### （1）输出价格30元及以上商用大模型排行榜
| 大模型 |  输出价格  | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|-------|----------|---------|---------|--------|--------|---------|---------|-------|------|------|-------|--------|----------|---|----|-----|
|hunyuan-turbo☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|50元|93.0|85.2|89.4|                    97.3|78.0|99.5|93.7|83.2|                    92.0|82.4|82.6|69.1|                    90.6|85.9|1|
|ERNIE-4.0☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|90元|88.0|89.0|93.8|                    94.0|79.0|100.0|88.6|82.8|                    92.0|84.0|76.0|61.0|                    83.7|84.8|2|
|ERNIE-4.0-Turbo-8K☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|60元|90.0|94.8|93.2|                    98.7|78.0|97.7|82.9|82.8|                    92.7|86.4|71.7|58.6|                    81.0|84.1|3|
|GLM-4-Plus☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|50元|87.0|91.9|90.2|                    99.3|81.0|88.7|89.5|87.0|                    90.9|89.4|76.7|56.8|                    86.9|84.1|4|
|xunfei-4.0Ultra☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|100元|88.0|84.4|94.0|                    92.7|80.0|94.3|93.7|81.9|                    92.0|85.0|72.0|62.0|                    83.1|82.9|5|
|Claude-3.5-Sonnet☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|108.6元|97.0|94.8|84.6|                    99.3|81.8|92.2|82.7|91.1|                    95.1|86.1|64.0|42.0|                    73.9|82.2|6|
|qwen-max☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|60元|92.0|88.9|91.5|                    99.3|77.0|79.8|91.9|74.5|                    93.0|88.9|73.6|47.0|                    84.5|81.9|7|
|SenseChat-5☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|100元|93.0|90.4|87.2|                    97.3|82.0|85.0|82.9|86.2|                    90.0|86.0|70.0|45.0|                    74.8|81.4|8|
|360gpt2-o1☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|50元|98.0|94.1|77.4|                    100.0|78.8|90.4|91.5|85.5|                    89.2|83.8|70.5|48.0|                    79.9|81.2|9|
|xunfei-spark-max☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|30元|87.0|92.0|82.6|                    87.3|74.0|93.5|93.7|72.5|                    91.6|87.0|70.4|59.6|                    84.7|80.9|10|
|gemini-1.5-pro☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|36元|87.0|90.4|84.2|                    99.3|75.0|92.2|92.5|85.9|                    91.3|84.2|69.7|31.3|                    77.7|80.4|11|
|GLM4☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|100元|92.0|86.7|90.0|                    98.0|77.0|78.0|84.3|77.0|                    93.0|83.0|64.0|38.0|                    81.3|79.5|12|
|gpt-4o☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|72.4元|93.0|96.3|90.0|                    100.0|83.0|95.7|81.1|72.8|                    87.1|82.7|67.6|35.0|                    72.7|79.5|13|
|mistral-large☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|43.4元|88.0|97.0|79.3|                    97.3|74.6|93.7|88.7|89.5|                    91.3|82.6|66.5|33.5|                    69.8|79.3|14|
|xunfei-spark-pro☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|30元|87.0|82.0|86.2|                    86.0|74.0|94.0|94.6|35.0|                    90.9|86.9|60.8|63.0|                    78.4|76.8|15|
|Baichuan4☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|100元|86.0|94.1|89.2|                    95.3|75.0|78.2|75.1|82.3|                    90.0|83.0|62.0|34.4|                    71.6|74.5|16|

<br>

##### （2）输出价格5~30元商用大模型排行榜
| 大模型 |     输出价格                         | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|---|----|---|----|--|--|---|
|hunyuan-large☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|12元|91.0|88.9|90.8|                    96.7|79.0|93.0|93.9|88.9|                    92.7|81.6|86.3|79.3|                    86.1|87.3|1|
|360gpt2-pro☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|5元|99.0|91.9|87.9|                    100.0|83.9|96.0|92.2|89.2|                    89.8|87.0|72.7|49.6|                    77.9|83.7|2|
|360gpt-pro☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|5元|97.0|90.4|87.9|                    100.0|83.5|96.0|92.2|88.4|                    89.2|87.0|73.3|49.8|                    77.9|83.4|3|
|qwen2.5-72b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|12元|92.0|87.4|91.7|                    92.7|83.0|95.5|91.1|85.8|                    91.3|86.6|71.7|49.1|                    82.5|82.6|4|
|abab7-chat-preview☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|10元|89.0|96.3|83.9|                    97.3|83.0|94.2|86.1|82.4|                    92.3|87.8|74.0|48.4|                    75.5|82.1|5|
|qwen2.5-32b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|7元|91.0|94.1|85.8|                    91.3|83.0|94.0|90.3|66.6|                    94.1|88.2|70.0|51.9|                    81.2|81.1|6|
|step-1-8k☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|20元|96.0|93.3|81.1|                    100.0|83.1|94.2|84.5|88.1|                    90.9|83.0|69.1|45.4|                    70.3|80.7|7|
|Baichuan4-Turbo☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|15元|91.0|93.3|85.7|                    100.0|78.0|93.2|92.0|81.9|                    88.5|87.2|66.2|43.2|                    74.7|79.7|8|
|qwen2.5-14b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|6元|89.0|90.4|85.0|                    98.0|81.0|91.5|93.7|54.4|                    92.7|87.5|67.0|42.6|                    79.3|79.6|9|
|GLM-4-AirX☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|10元|89.0|91.9|85.7|                    88.0|83.0|74.2|84.0|57.7|                    88.9|83.7|72.2|45.9|                    78.5|77.5|10|
|Hermes-3-Llama-3.1-405B☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|5.8元|94.0|92.6|78.1|                    100.0|80.1|90.2|80.1|90.7|                    86.1|83.0|64.7|29.4|                    62.4|77.3|11|
|Meta-Llama-3.1-405B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|21元|90.0|90.4|76.5|                    98.7|76.7|95.0|64.2|91.0|                    88.9|79.7|64.2|37.4|                    60.4|76.5|12|
|moonshot-v1-8k☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|12元|92.0|85.0|78.0|                    89.3|72.0|79.3|85.1|66.7|                    86.4|82.9|62.5|34.2|                    75.2|74.8|13|
|SenseChat-Turbo☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|5元|81.0|77.8|83.9|                    86.0|72.0|78.5|81.9|74.1|                    89.9|82.9|63.9|41.5|                    72.4|74.4|14|
|SenseChat-v4☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|12元|89.0|78.5|75.1|                    86.7|71.0|72.2|39.0|70.7|                    84.7|76.8|53.3|25.2|                    55.5|67.7|15|
|gemini-1.0-pro☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|10.8元|84.0|89.6|74.1|                    99.3|76.0|50.8|40.6|75.0|                    67.6|76.3|49.2|24.2|                    54.0|65.8|16|

<br>

##### （3）输出价格1~5元商用大模型排行榜
| 大模型 |     输出价格                       | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|---|----|---|----|--|--|---|
|Doubao-pro-32k☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2元|86.0|88.1|88.7|                    86.7|85.0|98.2|91.0|84.3|                    92.0|88.1|76.3|56.6|                    89.5|83.1|1|
|360gpt-turbo☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2元|97.0|97.0|84.9|                    100.0|83.1|93.8|88.7|80.9|                    89.8|85.8|68.0|42.2|                    73.7|82.0|2|
|deepseek-chat-v3☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|2元|93.0|97.0|73.4|                    100.0|84.0|99.0|91.4|90.5|                    94.4|86.8|72.7|39.5|                    75.3|81.8|3|
|ERNIE-3.5-8K☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2元|94.0|89.6|87.6|                    100.0|72.0|100.0|81.8|68.8|                    91.3|86.2|71.1|57.1|                    80.9|81.6|4|
|qwen-plus☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2元|88.0|89.6|86.2|                    84.0|73.0|93.0|91.4|67.7|                    93.0|86.3|72.0|48.6|                    84.5|79.7|5|
|qwen-long☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2元|89.0|85.9|86.2|                    86.7|75.0|83.3|91.3|64.6|                    92.3|86.3|72.5|48.2|                    83.7|78.6|6|
|Llama-3.3-70B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|4.13元|97.0|94.8|77.1|                    99.3|80.9|93.5|75.8|90.1|                    87.5|79.4|66.4|29.4|                    61.3|77.6|7|
|gemini-1.5-flash☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2.2元|91.0|87.4|78.4|                    97.3|77.0|91.8|88.7|83.3|                    88.5|83.9|61.4|24.1|                    69.9|77.3|8|
|Llama-3.3-70B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|2.2元|93.0|96.3|76.3|                    100.0|83.5|94.2|70.5|89.9|                    87.1|77.2|64.8|28.5|                    60.3|76.9|9|
|Llama-3.1-Nemotron-70B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|2.2元|97.0|97.8|76.7|                    100.0|75.8|93.5|64.1|84.6|                    89.2|81.6|63.7|33.1|                    55.8|76.7|10|
|qwen2.5-7b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|2元|85.0|88.1|78.3|                    91.3|77.0|89.8|79.9|61.7|                    90.6|83.4|59.6|42.5|                    73.1|75.4|11|
|hunyuan-standard☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|2元|87.0|89.6|85.3|                    85.3|74.0|83.0|80.0|72.3|                    86.8|75.4|68.8|33.1|                    64.7|74.9|12|
|step-1-flash☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|4元|91.0|85.2|77.5|                    100.0|76.7|84.5|69.2|75.3|                    84.7|80.2|58.5|37.7|                    61.9|74.2|13|
|Meta-Llama-3.1-70B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|2.2元|92.0|93.3|75.6|                    97.3|76.3|94.2|59.8|86.5|                    88.9|79.8|59.1|29.6|                    54.8|74.0|14|
|Yi-1.5-34B-Chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|1.3元|90.0|83.0|81.8|                    83.3|74.0|79.0|75.6|77.2|                    84.0|81.3|59.0|38.9|                    67.8|73.9|15|
|gpt-4o-mini☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|4.3元|90.0|93.3|68.9|                    100.0|83.0|92.7|80.7|65.6|                    84.7|77.7|54.7|23.2|                    60.6|73.5|16|
|gemma-2-27b-it☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|1.26元|92.0|93.3|74.8|                    96.7|83.1|88.3|66.4|74.8|                    80.5|80.0|57.1|22.9|                    53.2|73.1|17|
|gemini-1.5-flash-8b☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|1.1元|93.0|90.4|70.0|                    99.3|84.7|77.3|81.3|71.7|                    79.1|79.6|51.6|19.6|                    57.8|72.5|18|
|Llama-3.1-70B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|4.1元|87.0|88.9|75.7|                    90.7|79.0|94.8|49.2|84.0|                    88.9|81.1|58.2|31.2|                    56.1|72.4|19|
|WizardLM-2-8x22B☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|3.6元|94.0|91.9|63.5|                    97.3|74.2|84.5|64.9|80.3|                    92.7|73.2|53.5|23.5|                    48.3|71.2|20|
|mistral-small☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|4.34元|91.0|91.1|62.5|                    96.7|65.3|89.5|76.3|76.9|                    90.6|79.7|51.0|21.4|                    48.6|71.1|21|
|Mixtral-8x7B-Instruct-v0.1☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|1.74元|92.0|86.7|60.9|                    90.0|64.4|69.8|46.8|63.0|                    81.2|73.0|47.8|18.1|                    42.6|63.0|22|

<br>

##### （4）输出价格1元以下商用大模型排行榜
| 大模型 |     输出价格                         | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|---|----|---|----|--|--|---|
|gemini-2.0-flash-exp☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0元|97.0|96.3|76.7|                    100.0|78.0|96.8|95.5|90.1|                    91.0|86.0|69.3|37.7|                    71.5|81.6|1|
|yi-lightning☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0.99元|94.0|90.4|79.8|                    100.0|82.0|96.0|83.5|82.4|                    90.6|84.7|69.0|41.1|                    77.2|80.8|2|
|internlm2_5-20b-chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|1元|86.0|90.4|79.7|                    97.3|75.0|89.7|86.8|78.7|                    88.2|82.2|66.4|42.7|                    74.1|77.7|3|
|phi-4☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|1元|96.0|93.3|70.7|                    97.3|75.0|97.2|86.1|86.1|                    91.6|80.6|66.1|23.6|                    58.8|77.4|4|
|abab6.5s-chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|1元|87.0|88.0|76.5|                    88.0|80.0|91.7|75.9|75.8|                    89.2|80.3|65.7|35.2|                    64.1|77.2|5|
|GLM-4-Long☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|1元|85.0|93.3|78.3|                    96.7|80.0|81.2|79.0|81.2|                    88.9|81.6|65.0|40.6|                    75.1|76.8|6|
|GLM-4-Air☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|1元|89.0|91.9|84.7|                    88.0|83.0|74.5|78.1|56.8|                    89.2|83.7|69.7|40.7|                    78.0|76.4|7|
|Baichuan4-Air☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0.98元|90.0|91.9|85.5|                    97.3|75.4|90.0|77.5|77.3|                    85.4|84.0|55.9|29.8|                    65.7|75.6|8|
|qwen-turbo☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0.6元|83.0|85.2|85.1|                    76.0|66.0|81.3|89.6|64.4|                    91.6|83.2|67.3|44.6|                    77.7|75.2|9|
|GLM-4-Flash☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0元|89.0|80.0|80.5|                    82.0|79.0|75.5|78.3|61.7|                    89.2|80.3|64.5|39.2|                    76.1|73.5|10|
|internlm2_5-7b-chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.4元|86.0|84.4|78.0|                    83.3|79.0|59.8|81.1|73.5|                    87.1|83.0|62.4|43.8|                    68.9|73.2|11|
|gemma-2-9b-it☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.6元|85.0|82.2|74.8|                    87.3|81.0|89.3|67.4|59.9|                    81.9|78.5|53.6|19.1|                    53.8|69.4|12|
|ERNIE-Speed-8K☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0元|88.0|88.1|77.2|                    89.3|68.0|68.7|65.7|54.1|                    86.4|80.5|54.5|30.8|                    62.2|68.8|13|
|ministral-8b☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0.72元|88.0|90.4|60.2|                    99.3|78.6|85.5|69.0|71.4|                    87.5|59.4|45.3|21.1|                    44.0|68.1|14|
|Yi-1.5-9B-Chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.4元|82.0|83.0|73.9|                    80.0|72.0|73.8|54.7|70.8|                    85.4|75.8|45.3|31.5|                    56.9|66.6|15|
|Mistral-Nemo-Instruct-2407☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.58元|89.0|91.9|58.0|                    93.3|75.0|79.3|52.4|69.9|                    81.9|75.2|42.4|20.9|                    48.1|65.9|16|
|Doubao-lite-32k☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0.6元|77.0|86.7|71.9|                    64.7|62.0|87.2|71.8|52.3|                    79.4|64.6|49.8|32.1|                    68.4|65.5|17|
|Llama-3.1-8B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.4元|63.0|85.2|65.3|                    84.0|69.0|90.5|50.4|65.7|                    71.8|77.9|49.6|22.2|                    44.6|63.9|18|
|Meta-Llama-3.1-8B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.36元|77.0|89.6|64.4|                    93.3|67.4|89.8|33.1|70.1|                    68.6|77.2|43.2|22.9|                    46.0|63.7|19|
|ministral-3b☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|0.29元|83.0|87.4|54.5|                    84.0|77.1|66.3|64.4|64.5|                    67.5|64.1|38.1|15.8|                    39.0|60.5|20|
|Llama-3.2-3B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.18元|74.0|83.0|58.1|                    88.7|74.6|89.7|46.2|58.1|                    63.4|69.6|37.8|18.4|                    35.3|60.2|21|
|Mistral-7B-Instruct-v0.3☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.4元|82.0|80.7|57.6|                    83.3|68.2|33.5|31.7|56.4|                    76.3|73.0|40.9|17.1|                    34.0|55.6|22|
|Llama-3.2-1B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|0.15元|56.0|71.9|52.3|                    54.0|61.0|67.3|23.3|22.2|                    56.1|53.0|32.7|14.4|                    33.7|45.9|23|

<br>

DIY自定义维度筛选榜单：☛ [link](https://easyllm.site/static/benchmarking.html) 

旗舰商用模型badcase: [gpt-4o](http://easyllm.site/static/badcase/badcase-of-llm.html?model=gpt-4o) | 
[deepseek-chat-v3](http://easyllm.site/static/badcase/badcase-of-llm.html?model=deepseek-chat-v3) |
[更多](http://easyllm.site/static/badcase.html)
<br><br>

#### 1.2、开源大模型排行榜
##### （1）5B以下开源大模型排行榜
| 类别 | 大模型   | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|----|----------|--------|---------|--------|---------|---------|---------|------|-------|------|-------|--------|----------|---|----|---|
|开源|qwen2.5-3b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|81.0|75.6|70.6|83.3|                        77.0|85.7|75.5|43.5|84.3|                        80.3|51.3|28.9|56.5|68.0|1|
|商用|ministral-3b☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|83.0|87.4|54.5|84.0|                        77.1|66.3|64.4|64.5|67.5|                        64.1|38.1|15.8|39.0|60.5|2|
|开源|Llama-3.2-3B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|74.0|83.0|58.1|88.7|                        74.6|89.7|46.2|58.1|63.4|                        69.6|37.8|18.4|35.3|60.2|3|
|开源|qwen2.5-1.5b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|70.0|71.9|65.6|63.3|                        62.0|83.3|56.1|34.0|36.2|                        75.1|40.5|28.1|52.5|55.9|4|
|开源|Llama-3.2-1B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|56.0|71.9|52.3|54.0|                        61.0|67.3|23.3|22.2|56.1|                        53.0|32.7|14.4|33.7|45.9|5|
|开源|qwen2.5-0.5b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|52.0|53.3|50.4|46.0|                        58.0|51.8|36.6|15.7|48.1|                        50.4|30.7|21.7|37.4|41.8|6|

<br>

##### （2）5B~20B开源大模型排行榜
| 类别 | 大模型                        | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|---|----|---|----|--|--|---|
|开源|qwen2.5-14b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|89.0|90.4|85.0|98.0|                        81.0|91.5|93.7|54.4|92.7|                        87.5|67.0|42.6|79.3|79.6|1|
|开源|internlm2_5-20b-chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|86.0|90.4|79.7|97.3|                        75.0|89.7|86.8|78.7|88.2|                        82.2|66.4|42.7|74.1|77.7|2|
|开源|phi-4☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|96.0|93.3|70.7|97.3|                        75.0|97.2|86.1|86.1|91.6|                        80.6|66.1|23.6|58.8|77.4|3|
|开源|qwen2.5-7b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|85.0|88.1|78.3|91.3|                        77.0|89.8|79.9|61.7|90.6|                        83.4|59.6|42.5|73.1|75.4|4|
|开源|internlm2_5-7b-chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|86.0|84.4|78.0|83.3|                        79.0|59.8|81.1|73.5|87.1|                        83.0|62.4|43.8|68.9|73.2|5|
|开源|glm-4-9b-chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|90.0|82.2|80.0|82.0|                        79.0|76.5|74.5|62.4|88.9|                        80.3|64.1|38.4|75.3|73.2|6|
|商用|gemini-1.5-flash-8b☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|93.0|90.4|70.0|99.3|                        84.7|77.3|81.3|71.7|79.1|                        79.6|51.6|19.6|57.8|72.5|7|
|开源|gemma-2-9b-it☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|85.0|82.2|74.8|87.3|                        81.0|89.3|67.4|59.9|81.9|                        78.5|53.6|19.1|53.8|69.4|8|
|商用|ministral-8b☛[去体验](https://easyllm.site/static/modelcompare.html?type=proprietary)|88.0|90.4|60.2|99.3|                        78.6|85.5|69.0|71.4|87.5|                        59.4|45.3|21.1|44.0|68.1|9|
|开源|Yi-1.5-9B-Chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|82.0|83.0|73.9|80.0|                        72.0|73.8|54.7|70.8|85.4|                        75.8|45.3|31.5|56.9|66.6|10|
|开源|Mistral-Nemo-Instruct-2407☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|89.0|91.9|58.0|93.3|                        75.0|79.3|52.4|69.9|81.9|                        75.2|42.4|20.9|48.1|65.9|11|
|开源|Llama-3.1-8B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|63.0|85.2|65.3|84.0|                        69.0|90.5|50.4|65.7|71.8|                        77.9|49.6|22.2|44.6|63.9|12|
|开源|Meta-Llama-3.1-8B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|77.0|89.6|64.4|93.3|                        67.4|89.8|33.1|70.1|68.6|                        77.2|43.2|22.9|46.0|63.7|13|
|开源|Mistral-7B-Instruct-v0.3☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|82.0|80.7|57.6|83.3|                        68.2|33.5|31.7|56.4|76.3|                        73.0|40.9|17.1|34.0|55.6|14|

<br>

##### （3）20B以上开源大模型排行榜
| 类别 | 大模型                        | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算 |初中数学|符号推理|代词理解|诗词匹配|公务员考试|律师资格考试|高考|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|---|----|---|----|--|--|---|
|开源|qwen2.5-72b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|92.0|87.4|91.7|92.7|                        83.0|95.5|91.1|85.8|91.3|                        86.6|71.7|49.1|82.5|82.6|1|
|开源|deepseek-chat-v3☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|93.0|97.0|73.4|100.0|                        84.0|99.0|91.4|90.5|94.4|                        86.8|72.7|39.5|75.3|81.8|2|
|开源|qwen2.5-32b-instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|91.0|94.1|85.8|91.3|                        83.0|94.0|90.3|66.6|94.1|                        88.2|70.0|51.9|81.2|81.1|3|
|开源|Llama-3.3-70B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|97.0|94.8|77.1|99.3|                        80.9|93.5|75.8|90.1|87.5|                        79.4|66.4|29.4|61.3|77.6|4|
|开源|Hermes-3-Llama-3.1-405B☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|94.0|92.6|78.1|100.0|                        80.1|90.2|80.1|90.7|86.1|                        83.0|64.7|29.4|62.4|77.3|5|
|开源|Llama-3.3-70B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|93.0|96.3|76.3|100.0|                        83.5|94.2|70.5|89.9|87.1|                        77.2|64.8|28.5|60.3|76.9|6|
|开源|Llama-3.1-Nemotron-70B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|97.0|97.8|76.7|100.0|                        75.8|93.5|64.1|84.6|89.2|                        81.6|63.7|33.1|55.8|76.7|7|
|开源|Meta-Llama-3.1-405B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|90.0|90.4|76.5|98.7|                        76.7|95.0|64.2|91.0|88.9|                        79.7|64.2|37.4|60.4|76.5|8|
|开源|Meta-Llama-3.1-70B-Instruct-fp8☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|92.0|93.3|75.6|97.3|                        76.3|94.2|59.8|86.5|88.9|                        79.8|59.1|29.6|54.8|74.0|9|
|开源|Yi-1.5-34B-Chat☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|90.0|83.0|81.8|83.3|                        74.0|79.0|75.6|77.2|84.0|                        81.3|59.0|38.9|67.8|73.9|10|
|开源|gemma-2-27b-it☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|92.0|93.3|74.8|96.7|                        83.1|88.3|66.4|74.8|80.5|                        80.0|57.1|22.9|53.2|73.1|11|
|开源|Llama-3.1-70B-Instruct☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|87.0|88.9|75.7|90.7|                        79.0|94.8|49.2|84.0|88.9|                        81.1|58.2|31.2|56.1|72.4|12|
|开源|WizardLM-2-8x22B☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|94.0|91.9|63.5|97.3|                        74.2|84.5|64.9|80.3|92.7|                        73.2|53.5|23.5|48.3|71.2|13|
|开源|Mixtral-8x7B-Instruct-v0.1☛[去体验](https://easyllm.site/static/modelcompare.html?type=open-source)|92.0|86.7|60.9|90.0|                        64.4|69.8|46.8|63.0|81.2|                        73.0|47.8|18.1|42.6|63.0|14|


DIY自定义维度筛选榜单：☛[link](https://easyllm.site/static/benchmarking.html)

<br><br>


### 2、【学科知识】高考排行榜
历年高考题，共1500多道，绝大部分为选择题，少部分为填空题，参考[AGIEval](https://github.com/ruixiangcui/AGIEval)
![lin](pic/gaokao.png)
<br>
#### （1）高考生物
评测样本举例：
> 已知(1)酶、(2)抗体、(3)激素、(4)糖原、(5)脂肪、(6)核酸都是人体内有重要作用的物质。下列说法正确的 是    
(A)(1)(2)(3)都是由氨基酸通过肽键连接而成的   
(B)(3)(4)(5)都是生物大分子, 都以碳链为骨架   
(C)(1)(2)(6)都是由含氮的单体连接成的多聚体   
(D)(4)(5)(6)都是人体细胞内的主要能源物质   
>     

![lin](pic/gaokao-biology.png)
☛查看[高考生物badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-biology)
<br><br>

#### （2）高考化学
评测样本举例：
> 以下是中华民族为人类文明进步做出巨大贡献的几个事例, 运用化学知识对其 进行的分析不合理的是 ( )   
(A)四千余年前用谷物酿造出酒和酯, 酿造过程中只发生水解反应   
(B)商代后期铸造出工艺精湛的后（司）母戊鼎, 该鼎属于铜合金制品   
(C)汉代烧制出“明如镜、声如磬”的瓷器，其主要原料为黏土   
(D)屠呦呦用乙醚从青蒿中提取出对治疗疘疾有特效的青高素, 该过程包括萃取操作    
>    

![lin](pic/gaokao-chemistry.png)
☛查看[高考化学badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-chemistry)
<br><br>

#### （3）高考语文
评测样本举例：
> 下列各句中，没有语病的一句是   
(A)根据本报和部分出版机构联合开展的调查显示，儿童的阅读启蒙集中在1~2岁之间，并且阅读时长是随着年龄的增长而增加的。   
(B)为了培养学生关心他人的美德，我们学校决定组织开展义工服务活动，三个月内要求每名学生完成20个小时的义工服务。   
(C)在互联网时代，各领域发展都需要速度更快、成本更低的信息网络，网络提速降费能够推动“互联网+”快速发展和企业广泛收益。   
(D)面对经济全球化带来的机遇和挑战，正确的选择是，充分利用一切机遇，合作一切挑战，引导好经济全球化走向。  
>   

![lin](pic/gaokao-chinese.png)
☛查看[高考语文badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-chinese)
<br><br>

#### （4）高考地理
评测样本举例：
> 农业生产中地膜覆盖对土壤理化性状的主要作用是（）   
①保持土壤温度  ②减少水肥流失  ③增加土壤厚度  ④改善土壤质地     
(A)①②    
(B)①④   
(C)②③   
(D)③④   
>    

![lin](pic/gaokao-geography.png)
☛查看[高考地理badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-geography)
<br><br>

#### （5）高考历史
评测样本举例：
> “一万年农业，五千年文明，两千年大一统”指的是  
(A)中华文明  
(B)埃及文明  
(C)印度文明  
(D)希腊文明   
>  

![lin](pic/gaokao-history.png)
☛查看[高考历史badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-history)
<br><br>

#### （6）高考数学
评测样本举例：
> 已知 a ∈ R, (1+a*i)i=3+i, (i为虚数单位), 则 a=()  
(A)-1 (B)1 (C)-3 (D)3    

![lin](pic/gaokao-math.png)
☛查看[高考数学badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-math)
<br><br>

#### （7）高考物理
评测样本举例：
> 20 世纪 60 年代, 我国以国防为主的尖端科技取得了突破性的发展。1964 年, 我国第一颗原子弹试爆成 功； 1967 年, 我国第一颗氢弹试爆成功。关于原子弹和氢弹, 下列说法正确的是（ ）    
(A)原子弹和氢弹都是根据核裂变原理研制的   
(B)原子弹和氢弹都是根据核聚变原理研制的   
(C)原子弹是根据核裂变原理研制的，氢弹是根据核聚变原理研制的   
(D)原子弹是根据核聚变原理研制的，氢弹是根据核裂变原理研制的   
>     

![lin](pic/gaokao-physics.png)
☛查看[高考物理badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=gaokao-physics)
<br><br><br>


### 3、【推理】常识推理排行榜
常识推理选择题，共99道，参考[ISP](https://arxiv.org/abs/2306.09479)。

评测样本举例：
> 以下是关于常识的选择题。   
问题：当某人把土豆放到篝火边的余烬中，此时余烬并没有在   
A、释放热量  
B、吸收热量   
>      

![lin](pic/commonsense.png)
☛查看[常识推理badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=commonsense)
<br><br><br>


### 4、【逻辑推理】公务员考试排行榜
公务员考试行测选择题，共651道，参考[AGIEval](https://github.com/ruixiangcui/AGIEval)。
评测样本举例：
> 某乡镇进行新区规划，决定以市民公园为中心，在东南西北分别建设一个特色社区。这四个社区分别定为，文化区、休闲区、商业区和行政服务区。已知行政服务区在文化区的西南方向，文化区在休闲区的东南方向。   
根据以上陈述，可以得出以下哪项？   
(A)市民公园在行政服务区的北面    
(B)休闲区在文化区的西南   
(C)文化区在商业区的东北   
(D)商业区在休闲区的东南   
>  

![lin](pic/kaogong.png)
☛查看[公务员考试badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=kaogong)
<br><br><br>


### 5、【推理】律师资格考试排行榜
律师资格考试选择题，共2000道，参考[AGIEval](https://github.com/ruixiangcui/AGIEval)。
评测样本举例：
> 中国商务部决定对原产于马来西亚等八国的橡胶制品展开反补贴调查。根据我国《反补贴条例》以及相关法律法规，下列关于此次反补贴调查的哪项判断是正确的?（请选择一个或多个选项）    
(A)我国商务部在确定进口橡胶制品是否存在补贴时必须证明出国(地区)政府直接向出口商提供了现金形式的财政资助   
(B)在反补贴调查期间，该八国政府或橡胶制品的出口经营者，可以向中国商务部作出承诺，取消、限制补贴或改变价格    
(C)如果我国商务部终局裁定决定对该八国进口橡胶制品征收反补贴税，该反补贴税的征收期限不得超过10年   
(D)如果中国橡胶制品进口商对商务部征收反补贴税的终局裁定不服，必须首先向商务部请求行政复审，对行政复审决定还不服，才能向中国有管辖权的法院起诉    
> 

![lin](pic/jecqa.png)
☛查看[律师资格考试（一）badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=JEC-QA-KD)
☛查看[律师资格考试（二）badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=JEC-QA-CA)
<br><br><br>


### 6、【推理】符号推理BBH排行榜
学术界最常用的符号推理评测集，包含23个子任务，详细介绍见[BBH](https://easyllm.site/static/benchmarks.html)。
评测样本举例：
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
(A) 1pm to 2pm   
(B) 6pm to 7pm   
(C) 5pm to 6pm   
(D) 2pm to 4pm   
A:    
> 

完整排行榜见[BBH](leaderboard/bbh.md)<br>
☛查看[BBH符号推理badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=BBH)
<br><br><br>


### 7、【数学计算】初中数学排行榜
七/八/九年级的平均分计入总分。<br>
评分标准：七、八、九年级分别有40道题、21道题、36道题，所有题目都只判断对错（没有中间分数）。对于任何题目，只有模型response完全正确才给分，部分正确或错误都不得分。<br>
评测样本举例：
> 因式分解：3x^2y-12xy+12y

![lin](pic/Grade7Math-zh.png)
☛查看[七年级数学badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=Grade7Math-zh)

![lin](pic/Grade8Math-zh.png)
☛查看[八年级数学badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=Grade8Math-zh)

![lin](pic/Grade9Math-zh.png)
☛查看[九年级数学badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=Grade9Math-zh)
<br><br><br>


### 8、【数学计算】算术能力排行榜
考查大模型的数学基础能力之算数能力，测试题目为1000以内的整数加减法、不超过2位有效数字的浮点数加减乘除。
举例：166 + 215 + 53 = ？，0.97 + 0.4 / 4.51 = ？

完整排行榜见[arithmetic](leaderboard/arithmetic.md)<br>
☛查看[算术能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=arithmetic)
<br><br><br>


### 9、【语言理解】文本蕴含排行榜
文本蕴含，判断两个句子之间的语义关系：蕴含、中立、矛盾，参考[OCNLI](https://arxiv.org/abs/2010.05444)。

评测样本举例：
> 句子一：农机具购置补贴覆盖到全国所有农牧业县(场),中央财政拟安排资金130亿元,比上年增加90亿元   
句子二：按农民人数发放补贴  
以上两个句子是什么关系？   
(A)蕴含  
(B)中立  
(C)矛盾   
>   

![lin](pic/textEntail.png)
☛查看[文本蕴含badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=textEntail)
<br><br><br>


### 10、【语言理解】代词理解CLUEWSC排行榜
中文指代消解任务，参考[CLUEWSC2020](https://github.com/CLUEbenchmark/CLUEWSC2020)。
评测样本举例：
> 少平仍然不知道怎样给奶奶说清他姐夫的事，就只好随口说：“他犯了点错误，人家让他劳教！”  
上述文本中的“他犯了点错误”中的“他”是指少平吗？   
选项：(A)是   
(B)否      
>    

完整排行榜见[CLUEWSC](leaderboard/CLUEWSC.md)<br>
☛查看[代词理解CLUEWSC badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=CLUEWSC)
<br><br><br>


### 11、【语言理解】分类能力排行榜
评测样本举例：
> 将下列单词按词性分类。    
> 狗，追，跑，大人，高兴，树

完整排行榜见[classification](leaderboard/classification.md)<br>
☛查看[分类能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=classification)
<br><br><br>


### 12、【语言理解】信息抽取能力排行榜
评测样本举例：  
> “中信银行3亿元，交通银行增长约2.7亿元，光大银行约1亿元。”    
> 提取出以上文本中的所有组织机构名称

完整排行榜见[extract](leaderboard/info-extract.md)<br>
☛查看[信息抽取能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=extract)
<br><br><br>


### 13、【语言理解】阅读理解能力排行榜
阅读理解能力是一种符合能力，考查针对给定信息的理解能力。
依据给定信息的种类，可以细分为：文章问答、表格问答、对话问答……    
评测样本举例：
> 牙医：好的，让我们看看你的牙齿。从你的描述和我们的检查结果来看，你可能有一些牙齦疾病，导致牙齿的神经受到刺激，引起了敏感。此外，这些黑色斑点可能是蛀牙。  
病人：哦，真的吗？那我该怎么办？   
牙医：别担心，我们可以为你制定一个治疗计划。我们需要首先治疗牙龈疾病，然后清除蛀牙并填充牙洞。在此过程中，我们将确保您感到舒适，并使用先进的技术和材料来实现最佳效果。   
病人：好的，谢谢您，医生。那么我什么时候可以开始治疗？   
牙医：让我们为您安排一个约会。您的治疗将在两天后开始。在此期间，请继续刷牙，使用牙线，并避免吃过于甜腻和酸性的食物和饮料。   
病人：好的，我会的。再次感谢您，医生。   
牙医：不用谢，我们会尽最大的努力帮助您恢复健康的牙齿。   
基于以上对话回答：病人在检查中发现的牙齿问题有哪些？
> 

完整排行榜见[mrc](leaderboard/mrc.md)<br>
☛查看[阅读理解能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=mrc)
<br><br><br>


### 14、【传统文化】诗词匹配CCPM排行榜
中国古典诗歌匹配，给定中国古典诗歌的现代问描述，要求从候选的四句诗中选出与现代文描述语义匹配的那一句。
利用古典诗歌和现代文翻译的平行语料构建正确选项，并利用正确选项从古代诗歌语料库中利用相似检索构造出错误候选。
参考[CCPM](https://github.com/THUNLP-AIPoet/CCPM)。
评测样本举例：
> 昏暗的灯熄灭了又被重新点亮。   
上述文本最匹配下面哪句诗：   
(A)渔灯灭复明   
(B)残灯灭又然   
(C)残灯暗复明   
(D)残灯灭又明   
>    

完整排行榜见[CCPM](leaderboard/CCPM.md)<br>
☛查看[诗词匹配CCPM badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=CCPM)
<br><br><br>


### 15、数据分析排行榜
专门考查大模型对表格的理解分析能力，常用于数据分析。    
评测样本举例：
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

完整排行榜见[tableqa](leaderboard/table-qa.md)<br>
☛查看[数据分析badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=tableqa)
<br><br><br>


### 16、中文指令遵从排行榜
参考谷歌IFEval，并将其翻译和适配到中文，精选9类25种指令，说明如下：
![lin](pic/IFEval.jpg)

完整排行榜见[IFEval](leaderboard/IFEval.md)<br>
☛查看[中文指令遵从badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=IFEval-zh)
<br><br><br>


### 17、中文编码效率排行榜
暂不计入综合能力评分。
专门考查大模型编码中文字符的效率，同等尺寸大模型，编码效率越高推理速度越快，几乎成正比。
中文编码效率相当于大模型生成的每个token解码后对应的中文平均字数
（大模型每次生成一个token，然后解码成真正可见的字符，比如中文、英文、标点符号等）。
比如baichuan2、llama2的中文中文编码效率分别为1.67、0.61，意味着在同尺寸模型下，baichuan2的运行速度是llama2的2.7倍（1.67/0.61）。
![lin](pic/zhcoding.png)
<br><br><br>


## 🌐各项能力评分
评分方法：从各个维度给大模型打分，每个维度都对应一个评测数据集，包含若干道题。
每道题依据大模型回复质量给1~5分，将评测集内所有题的得分累加并归一化为100分制，即作为最终得分。

所有评分数据详见[alldata](leaderboard/alldata.md)
<br><br>


## ⚖️原始评测数据
包含各维度评测集以及大模型输出结果，详见本项目的[eval文件目录](eval)
<br><br>


## 为什么做榜单？
- 大模型百花齐放，也参差不齐。不少媒体的宣传往往夸大其词，避重就轻，容易混淆视听；而某些公司为了PR，也过分标榜自己大模型的能力，动不动就“达到chatgpt水平”，动不动就“国内第一”。
所谓“外行看热闹，内行看门道”，业界急需一股气流，摒弃浮躁，静下心来打磨前沿技术，真真正正用技术实力说话。这就少不了一个公开、公正、公平的大模型评测系统，把各类大模型的优点、不足一一展示出来。
如此，大家既能把握当下的发展水平、与国外顶尖技术的差距，也能更加清晰地看明白未来的努力方向，而不被资本热潮、舆论热潮所裹挟。
- 对于产业界来说，特别是对于不具备大模型研发能力的公司，熟悉大模型的技术边界、高效有针对性地做大模型技术选型，在现如今显得尤为重要。
而一个公开、公正、公平的大模型评测系统，恰好能够提供应有的助力，避免重复造轮子，避免因技术栈不同而导致不必要的争论，避免“鸡同鸭讲”。
- 对于大模型研发人员，包括对大模型技术感兴趣的人、学术界看中实践的人，各类大模型的效果对比，反应出了背后不同技术路线、技术方法的有效性，这就提供了非常好的参考意义。
不同大模型的相互参考、借鉴，帮忙大家躲过不必要的坑、避免重复实验带来的资源浪费，有助于整个大模型生态圈的良性高效发展。

## 大模型选型及评测交流群
先加小编微信，后拉入群，备注“加群”<br>
![lin](pic/qrcode-wxgroup.jpg)
<br><br><br><br>
关注大模型评测微信公众号，及时获取最新评测信息<br>
![lin](pic/qrcode-gzh.jpg)
