
# CLiB中文大模型能力评测榜单（持续更新）
- 目前已囊括128个大模型，覆盖chatgpt、gpt-4o、谷歌gemini、百度文心一言、阿里通义千问、百川、讯飞星火、商汤senseChat、minimax等商用模型，
以及qwen2.5、llama3.1、glm4、书生internLM2.5、openbuddy、AquilaChat等开源大模型。
- 模型来源涉及国内外大厂、大模型创业公司、高校研究机构。
- 支持多维度能力评测，包括分类能力、信息抽取能力、阅读理解能力、数据分析能力、中文编码效率、中文指令遵从、算术能力。
- 不仅提供能力评分排行榜，也提供所有模型的原始输出结果！有兴趣的朋友可以自己打分、自己排行！

## 目录
- [🔄最近更新](#最近更新)
- [⚓TODO](#todo)
- [📝大模型基本信息](#大模型基本信息)
- [📊排行榜](#-排行榜)
  - [综合能力排行榜](#1综合能力排行榜)
    - 10B以下开源大模型排行榜
    - 10B~20B开源大模型排行榜
    - 20B以上开源大模型排行榜
  - [分类能力排行榜](#2分类能力排行榜)
  - [信息抽取能力排行榜](#3信息抽取能力排行榜)
  - [阅读理解能力排行榜](#4阅读理解能力排行榜)
  - [数据分析排行榜](#5数据分析排行榜)
  - [中文指令遵从排行榜](#6中文指令遵从排行榜)
  - [数学基础（算术）能力排行榜](#7数学基础（算术）能力排行榜)
  - [中文编码效率排行榜](#8中文编码效率排行榜)
- [🌐各项能力评分](#🌐各项能力评分)
- [⚖️原始评测数据](#⚖️原始评测数据)
- [为什么做榜单？](#为什么做榜单)


## 最近更新
- [2024/10/20] 发布v2.3版本评测榜单
  - 新增6个模型：yi-lightning、gemini-1.5-flash、gemini-1.0-pro、gemini-1.5-pro、GLM-4-Long、GLM-4-Plus
  - 更新4个模型：GLM4、qwen-max、ERNIE-4.0-Turbo-8K、ERNIE-3.5-8K
  - 删除陈旧的模型：Baichuan2-13B-Chat、Baichuan2-7B-Chat、deepseek-llm-67b-chat、gpt4、gemma-2b-it、gemma-7b-it
- [2024/9/29]v2.2版本，[2024/8/27]v2.1版本，[2024/8/7]v2.0版本，[2024/7/26]v1.21版本，[2024/7/15]v1.20版本，[2024/6/29]v1.19版本，[2024/6/2]v1.18版本，[2024/5/8]v1.17版本，[2024/4/13]v1.16版本，[2024/3/20]v1.15版本，[2024/2/28]v1.14版本，[2024/1/29]v1.13版本
- 2023年：[2023/12/10]v1.12版本，[2023/11/22]v1.11版本，[2023/11/5]v1.10版本，[2023/10/11]v1.9版本，[2023/9/13]v1.8版本，[2023/8/29]v1.7版本，[2023/8/13]v1.6版本，[2023/7/26]v1.5版本， [2023/7/18]v1.4版本， [2023/7/2]v1.3版本， [2023/6/17]v1.2版， [2023/6/10]v1.1版本， [2023/6/4]v1版本

各版本更新详情：[CHANGELOG](CHANGELOG.md)

## TODO
- 将更多大模型加入评测：Claude等等
- 增加开源大模型的授权协议，注明能否商用
- 引入更多维度的评测：数学能力、代码能力、开放域问答、多轮对话、头脑风暴、翻译……
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
综合能力得分为分类能力、信息抽取、阅读理解、数据分析、指令遵从、算术运算六者得分的平均值。
![lin](pic/total.png)    
详细数据见[total](total.md)
<br>

#### 1.1、商用大模型排行榜（含开源模型的付费API）
##### （1）输出价格30元及以上商用大模型排行榜
| 大模型 | 价格（输出）                          | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|------------------------------|------|--------|--------|------|------|----|------|----|
|gpt-4o|72.4元|93|96.3|98.0|100.0|83|95.7|94.3|1|
|百度ERNIE-4.0-Turbo|60元|90|94.8|96.0|98.7|78|97.7|92.5|2|
|gpt-4-turbo|217元|91|90.0|94.0|96.0|83|96.5|91.8|3|
|百度ERNIE-4.0|90元|88|89.0|94.7|94.0|79|100.0|90.8|4|
|GLM-4-Plus(new)|50元|87|91.9|95.3|99.3|81|88.7|90.5|5|
|gemini-1.5-pro(new)|36元|87|90.4|93.3|99.3|75|92.2|89.5|6|
|讯飞4.0Ultra|100元|88|84.4|96.0|92.7|80|94.3|89.2|7|
|阿里qwen-max|60元|92|88.9|94.7|99.3|77|79.8|88.6|8|
|minimax-abab6.5-chat|30元|89|87.0|89.3|95.3|76|90.3|87.8|9|
|讯飞星火v3.5(spark-max)|30元|87|92.0|89.3|87.3|74|93.5|87.2|10|
|Baichuan4|100元|86|94.1|93.3|95.3|75|78.2|87.0|11|
|智谱GLM4|100元|92|86.7|90.0|98.0|77|78.0|87.0|12|
|讯飞星火v3(spark-pro)|30元|87|82.0|88.0|86.0|74|94.0|85.2|13|

<br>

##### （2）输出价格5~30元商用大模型排行榜
| 大模型 | 价格（输出）                          | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|------------------------------|------|--------|--------|------|------|----|------|----|
|qwen2.5-32b-instruct|7元|91|94.1|96.0|91.3|83|94.0|91.6|1|
|qwen2.5-14b-instruct|6元|89|90.4|94.0|98.0|81|91.5|90.7|2|
|Qwen2-72B-Instruct|12元|87|91.1|94.7|90.0|86|94.2|90.5|3|
|qwen2.5-72b-instruct|12元|92|87.4|92.0|92.7|83|95.5|90.4|4|
|Baichuan3-Turbo|12元|88|86.7|94.7|90.7|75|89.2|87.4|5|
|yi-large|20元|85|91.0|90.0|92.7|77|88.3|87.3|6|
|minimax-abab6.5s-chat|10元|87|88.0|88.7|88.0|80|91.7|87.2|7|
|智谱GLM-4-AirX|10元|89|91.9|92.7|88.0|83|74.2|86.5|8|
|qwen2-57b-a14b-instruct|7元|85|88.1|89.3|87.3|77|89.2|86.0|9|
|Qwen1.5-32B-Chat|7元|91|86.0|92.7|87.3|72|86.8|86.0|10|
|yi-large-turbo|12元|82|90.0|88.7|86.7|78|87.8|85.5|11|
|gpt-3.5-turbo|11元|81|83.0|92.7|91.3|77|80.0|84.2|12|
|Qwen1.5-72B-Chat|10元|89|84.0|88.0|87.3|70|84.8|83.8|13|
|月之暗面moonshot-v1-8k|12元|92|85.0|84.0|89.3|72|79.3|83.6|14|
|gemini-1.0-pro(new)|10.8元|84|89.6|92.7|99.3|76|50.8|82.1|15|
|商汤SenseChat-v4|12元|89|78.5|88.0|86.7|71|72.2|80.9|16|
|商汤SenseChat-Turbo|5元|81|77.8|76.7|86.0|72|78.5|78.7|17|
|minimax-abab5.5-chat|15元|83|79.0|86.7|72.7|76|39.7|72.8|18|
|minimax-abab5.5s-chat|5元|58|57.0|70.7|56.0|49|57.0|58.0|19|

<br>

##### （3）输出价格1~5元以下商用大模型排行榜
| 大模型 | 价格（输出）                          | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|------------------------------|------|--------|--------|------|------|----|------|----|
|百度ERNIE-3.5-8K|2元|94|89.6|98.0|100.0|72|100.0|92.3|1|
|gpt-4o-mini|4.3元|90|93.3|89.3|100.0|83|92.7|91.4|2|
|deepseek-chat-v2|2元|93|88.0|94.0|96.0|76|96.7|90.6|3|
|豆包Doubao-pro-32k|2元|86|88.1|96.7|86.7|85|98.2|90.1|4|
|gemini-1.5-flash(new)|2.2元|91|87.4|92.7|97.3|77|91.8|89.5|5|
|Llama-3.1-70B-Instruct|4.1元|87|88.9|92.0|90.7|79|94.8|88.7|6|
|yi-medium|2.5元|86|93.0|89.3|94.0|76|89.2|87.9|7|
|Llama-3-70B-Instruct|4.1元|88|87.0|96.0|95.0|70|90.8|87.8|8|
|qwen2.5-7b-instruct|2元|85|88.1|93.3|91.3|77|89.8|87.4|9|
|阿里qwen-plus|2元|88|89.6|90.0|84.0|73|93.0|86.3|10|
|阿里qwen-long|2元|89|85.9|90.0|86.7|75|83.3|85.0|11|
|Qwen2-7B-Instruct|2元|89|83.7|86.7|75.3|77|81.3|82.2|12|
|Qwen1.5-14B-Chat|4元|89|79.0|90.7|90.7|66|77.5|82.2|13|
|Yi-1.5-34B-Chat|1.3元|90|83.0|82.7|83.3|74|79.0|82.0|14|
|Qwen1.5-7B-Chat|2元|80|76.0|76.0|70.7|67|71.2|73.5|15|

<br>

##### （4）输出价格1元以下商用大模型排行榜
| 大模型 | 价格（输出）                          | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|------------------------------|------|--------|--------|------|------|----|------|----|
|yi-lightning(new)|0.99元|94|90.4|95.3|100.0|82|96.0|93.0|1|
|GLM-4-Long(new)|1元|85|93.3|89.3|96.7|80|81.2|87.6|2|
|internlm2_5-20b-chat|1元|86|90.4|86.0|97.3|75|89.7|87.4|3|
|智谱GLM-4-Air|1元|89|91.9|92.7|88.0|83|74.5|86.5|4|
|gemma-2-9b-it|0.6元|85|82.2|88.7|87.3|81|89.3|85.6|5|
|智谱GLM-4-Flash|0元|89|80.0|86.0|82.0|79|75.5|81.9|6|
|yi-spark|1元|82|88.9|88.0|76.0|72|83.3|81.7|7|
|百度ERNIE-Speed-8K|0元|88|88.1|88.0|89.3|68|68.7|81.7|8|
|Llama-3-8B-Instruct|0.4元|86|74.0|80.0|90.0|63|89.5|80.4|9|
|internlm2_5-7b-chat|0.4元|86|84.4|90.0|83.3|79|59.8|80.4|10|
|阿里qwen-turbo|0.6元|83|85.2|88.0|76.0|66|81.3|79.9|11|
|Yi-1.5-9B-Chat|0.4元|82|83.0|84.7|80.0|72|73.8|79.2|12|
|Llama-3.1-8B-Instruct|0.4元|63|85.2|82.0|84.0|69|90.5|79.0|13|
|豆包Doubao-lite-32k|0.6元|77|86.7|88.7|64.7|62|87.2|77.7|14|

<br>

旗舰商用模型badcase: [gpt-4o](http://easyllm.site/static/badcase/badcase-of-llm.html?model=gpt-4o) | 
[moonshot-v1-8k](http://easyllm.site/static/badcase/badcase-of-llm.html?model=moonshot-v1-8k) |
[deepseek-chat-v2](http://easyllm.site/static/badcase/badcase-of-llm.html?model=deepseek-chat-v2) |
[yi-large](http://easyllm.site/static/badcase/badcase-of-llm.html?model=yi-large) |
[更多](http://easyllm.site/static/badcase.html)
<br><br>

#### 1.2、开源大模型排行榜
##### （1）5B以下开源大模型排行榜
| 类别 | 大模型                        | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|
|开源|qwen2.5-3b-instruct|81|75.6|78.7|83.3|77|85.7|80.2|1|
|开源|qwen2.5-1.5b-instruct|70|71.9|72.7|63.3|62|83.3|70.5|2|
|开源|Phi-3-mini-128k-instruct|74|63.0|65.3|73.0|75|71.3|70.3|3|
|开源|MiniCPM-2B-dpo|79|77.0|74.0|66.0|55|52.7|67.3|4|
|开源|Qwen1.5-4B-Chat|75|65.0|79.3|63.0|56|53.0|65.2|5|
|开源|qwen2-1.5b-instruct|73|74.1|68.0|50.7|54|55.7|62.6|6|
|开源|qwen2.5-0.5b-instruct|52|53.3|63.3|46.0|58|51.8|54.1|7|
|开源|internlm2-chat-1_8b|69|60.7|63.3|46.0|45|39.7|54.0|8|
|开源|Qwen1.5-1.8B-Chat|57|58.0|52.7|48.0|46|26.7|48.1|9|
|开源|qwen2-0.5b-instruct|49|53.3|62.0|36.7|48|35.5|47.4|10|
|开源|Qwen1.5-0.5B-Chat|44|40.0|60.0|34.7|42|17.2|39.6|11|

<br>

##### （2）5B~10B开源大模型排行榜
| 类别 | 大模型                        | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|
|开源|qwen2.5-7b-instruct|85|88.1|93.3|91.3|77|89.8|87.4|1|
|开源|gemma-2-9b-it|85|82.2|88.7|87.3|81|89.3|85.6|2|
|开源|glm-4-9b-chat|90|82.2|90.0|82.0|79|76.5|83.3|3|
|开源|Qwen2-7B-Instruct|89|83.7|86.7|75.3|77|81.3|82.2|4|
|开源|Llama-3-8B-Instruct|86|74.0|80.0|90.0|63|89.5|80.4|5|
|开源|internlm2_5-7b-chat|86|84.4|90.0|83.3|79|59.8|80.4|6|
|开源|Yi-1.5-9B-Chat|82|83.0|84.7|80.0|72|73.8|79.2|7|
|开源|Llama-3.1-8B-Instruct|63|85.2|82.0|84.0|69|90.5|79.0|8|
|开源|openbuddy-llama3-8b|78|86.0|81.3|79.0|70|63.2|76.2|9|
|开源|Qwen1.5-7B-Chat|80|76.0|76.0|70.7|67|71.2|73.5|10|
|开源|internlm2-chat-7b|86|81.0|72.7|82.7|64|42.8|71.5|11|

<br>

##### （3）10B~20B开源大模型排行榜
| 类别 | 大模型                        | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|
|开源|qwen2.5-14b-instruct|89|90.4|94.0|98.0|81|91.5|90.7|1|
|开源|internlm2_5-20b-chat|86|90.4|86.0|97.3|75|89.7|87.4|2|
|开源|Qwen1.5-14B-Chat|89|79.0|90.7|90.7|66|77.5|82.2|3|
|开源|internlm2-chat-20b|93|80.0|86.0|88.0|68|63.3|79.7|4|
|开源|DeepSeek-V2-Lite-Chat|81|76.3|81.3|73.3|69|61.2|73.7|5|

<br>

##### （4）30B以上开源大模型排行榜
| 类别 | 大模型                        | 分类能力 | 信息抽取 | 阅读理解 | 数据分析 | 指令遵从 | 算术运算|总分   | 排名 |
|----|----------------------------|------|--------|--------|------|------|----|------|----|
|开源|qwen2.5-32b-instruct|91|94.1|96.0|91.3|83|94.0|91.6|1|
|开源|deepseek-chat-v2|93|88.0|94.0|96.0|76|96.7|90.6|2|
|开源|Qwen2-72B-Instruct|87|91.1|94.7|90.0|86|94.2|90.5|3|
|开源|qwen2.5-72b-instruct|92|87.4|92.0|92.7|83|95.5|90.4|4|
|开源|Llama-3.1-70B-Instruct|87|88.9|92.0|90.7|79|94.8|88.7|5|
|开源|Llama-3-70B-Instruct|88|87.0|96.0|95.0|70|90.8|87.8|6|
|开源|qwen2-57b-a14b-instruct|85|88.1|89.3|87.3|77|89.2|86.0|7|
|开源|Qwen1.5-32B-Chat|91|86.0|92.7|87.3|72|86.8|86.0|8|
|开源|Qwen1.5-72B-Chat|89|84.0|88.0|87.3|70|84.8|83.8|9|
|开源|Yi-1.5-34B-Chat|90|83.0|82.7|83.3|74|79.0|82.0|10|

<br>

### 2、分类能力排行榜
评测样本举例：
> 将下列单词按词性分类。    
> 狗，追，跑，大人，高兴，树

完整排行榜见[classification](classification.md)<br>
☛查看[分类能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=classification)
<br><br>

### 3、信息抽取能力排行榜
评测样本举例：  
> “中信银行3亿元，交通银行增长约2.7亿元，光大银行约1亿元。”    
> 提取出以上文本中的所有组织机构名称

完整排行榜见[extract](info-extract.md)<br>
☛查看[信息抽取能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=extract)
<br><br>

### 4、阅读理解能力排行榜
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

完整排行榜见[mrc](mrc.md)<br>
☛查看[阅读理解能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=mrc)
<br><br>

### 5、数据分析排行榜
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

完整排行榜见[tableqa](table-qa.md)<br>
☛查看[数据分析badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=tableqa)
<br><br>

### 6、中文指令遵从排行榜
参考谷歌IFEval，并将其翻译和适配到中文，精选9类25种指令，说明如下：
![lin](pic/IFEval.jpg)

完整排行榜见[IFEval](IFEval.md)<br>
☛查看[中文指令遵从badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=IFEval-zh)
<br><br>

### 7、数学基础（算术）能力排行榜
考查大模型的数学基础能力之算数能力，测试题目为1000以内的整数加减法、不超过2位有效数字的浮点数加减乘除。
举例：166 + 215 + 53 = ？，0.97 + 0.4 / 4.51 = ？

完整排行榜见[math](math.md)<br>
☛查看[算术能力badcase](http://easyllm.site/static/badcase/badcase-of-benchmark.html?benchmark=arithmetic)
<br><br>

### 8、中文编码效率排行榜
暂不计入综合能力评分。
专门考查大模型编码中文字符的效率，同等尺寸大模型，编码效率越高推理速度越快，几乎成正比。
中文编码效率相当于大模型生成的每个token解码后对应的中文平均字数
（大模型每次生成一个token，然后解码成真正可见的字符，比如中文、英文、标点符号等）。
比如baichuan2、llama2的中文中文编码效率分别为1.67、0.61，意味着在同尺寸模型下，baichuan2的运行速度是llama2的2.7倍（1.67/0.61）。
![lin](pic/zhcoding.png)
<br><br>

## 🌐各项能力评分
评分方法：从各个维度给大模型打分，每个维度都对应一个评测数据集，包含若干道题。
每道题依据大模型回复质量给1~5分，将评测集内所有题的得分累加并归一化为100分制，即作为最终得分。

所有评分数据详见[alldata](alldata.md)
<br>

## ⚖️原始评测数据
包含各维度评测集以及大模型输出结果，详见本项目的[eval文件目录](eval)


## 为什么做榜单？
- 大模型百花齐放，也参差不齐。不少媒体的宣传往往夸大其词，避重就轻，容易混淆视听；而某些公司为了PR，也过分标榜自己大模型的能力，动不动就“达到chatgpt水平”，动不动就“国内第一”。
所谓“外行看热闹，内行看门道”，业界急需一股气流，摒弃浮躁，静下心来打磨前沿技术，真真正正用技术实力说话。这就少不了一个公开、公正、公平的大模型评测系统，把各类大模型的优点、不足一一展示出来。
如此，大家既能把握当下的发展水平、与国外顶尖技术的差距，也能更加清晰地看明白未来的努力方向，而不被资本热潮、舆论热潮所裹挟。
- 对于产业界来说，特别是对于不具备大模型研发能力的公司，熟悉大模型的技术边界、高效有针对性地做大模型技术选型，在现如今显得尤为重要。
而一个公开、公正、公平的大模型评测系统，恰好能够提供应有的助力，避免重复造轮子，避免因技术栈不同而导致不必要的争论，避免“鸡同鸭讲”。
- 对于大模型研发人员，包括对大模型技术感兴趣的人、学术界看中实践的人，各类大模型的效果对比，反应出了背后不同技术路线、技术方法的有效性，这就提供了非常好的参考意义。
不同大模型的相互参考、借鉴，帮忙大家躲过不必要的坑、避免重复实验带来的资源浪费，有助于整个大模型生态圈的良性高效发展。

## 大模型选型及评测交流群
先加小编微信，后拉入群
![lin](pic/qrcode-wxgroup.jpg)
