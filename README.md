# 中文大模型能力评测榜单（持续更新）
- 目前已囊括14个大模型，覆盖chatgpt、百度文心一言、阿里通义千问、讯飞星火、minimax等商用模型，
以及belle、chatglm6b、ziya、guanaco、Phoenix等开源大模型。
- 模型来源涉及国内外大厂、大模型创业公司、高校研究机构。
- 支持多维度能力评测，包括分类能力、信息抽取能力、阅读理解能力。
- 不仅提供能力评分排行榜，也提供所有模型的原始输出结果！

## 🔄 最近更新
- [2023/6/10] 发布v1.1版本评测榜单
  - 新增3个大模型：minimax、guanaco、Phoenix-7b
  - 新增表格问答评测维度，作为阅读理解能力的细分项
- [2023/6/4] 发布v1版本评测榜单

## ⚓TODO
- 将更多大模型加入评测：gpt4、Claude、谷歌Bard、复旦moss、falcon、羊驼等等
- 引入更多维度的评测：数学能力、代码能力、开放域问答、多轮对话、头脑风暴、翻译……
- 加入更多评测数据，使得评测得分越来越有说服力

## 📝大模型基本信息
| 大模型                                                                    | 机构             | 类别 | 备注                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
|------------------------------------------------------------------------|----------------|----|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [chatgpt-3.5](https://chat.openai.com/)                                | openai         | 商用 | 风靡世界的AI产品，API为gpt3.5-turbo                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| [文心一言](https://yiyan.baidu.com/)                                       | 百度             | 商用 | 百度全新一代知识增强大语言模型，文心大模型家族的新成员，能够与人对话互动，回答问题，协助创作，高效便捷地帮助人们获取信息、知识和灵感。                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [chatglm官方](https://chatglm.cn/)                                       | 智谱AI           | 商用 | 一个具有问答、多轮对话和代码生成功能的中英双语模型，基于千亿基座 GLM-130B 开发，通过代码预训练、有监督微调等技术提升各项能力                                                                                                                                                                                                                                                                                                                                                                                                                 |
| [讯飞星火](https://xinghuo.xfyun.cn/desk)                                  | 科大讯飞           | 商用 | 具有文本生成、语言理解、知识问答、逻辑推理、数学能力、代码能力、多模态能力 7 大核心能力。该大模型目前已在教育、办公、车载、数字员工等多个行业和产品中落地。                                                                                                                                                                                                                                                                                                                                                                                                     |
| [阿里通义千问](https://tongyi.aliyun.com/)                                   | 阿里巴巴           | 商用 | 通义千问支持多轮对话，可进行文案创作、逻辑推理，支持多种语言。                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| [minimax](https://api.minimax.chat/)                                   | minimax        | 商用 | Glow app背后大模型                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| [chatglm-6b](https://github.com/THUDM/ChatGLM-6B)                      | 清华大学&智谱AI      | 开源 | ChatGLM-6B 是一个开源的、支持中英双语的对话语言模型，基于 General Language Model (GLM) 架构，具有 62 亿参数。结合模型量化技术，用户可以在消费级的显卡上进行本地部署（INT4 量化级别下最低只需 6GB 显存）。 ChatGLM-6B 使用了和 ChatGPT 相似的技术，针对中文问答和对话进行了优化。经过约 1T 标识符的中英双语训练，辅以监督微调、反馈自助、人类反馈强化学习等技术的加持，62 亿参数的 ChatGLM-6B 已经能生成相当符合人类偏好的回答                                                                                                                                                                                                                      |
| [belle-llama-7b-2m](https://github.com/LianjiaTech/BELLE)              | 链家科技           | 开源 | based on LLAMA 7B and finetuned with 2M Chinese data combined with 50,000 pieces of English data from the open source Stanford-Alpaca, resulting in good Chinese instruction understanding and response generation capabilities.                                                                                                                                                                                                                                                    |
| [BELLE-on-Open-Datasets](https://github.com/LianjiaTech/BELLE)         | 链家科技           | 开源 | Extending the vocabulary with additional 50K tokens specific for Chinese and further pretraining these word embeddings on Chinese corpus. Full-parameter finetuning the model with  instruction-following open datasets: alpaca, sharegpt, belle-3.5m.                                                                                                                                                                                                                              |
| [belle-llama-13b-2m](https://github.com/LianjiaTech/BELLE)             | 链家科技           | 开源 | based on LLAMA 13B and finetuned with 2M Chinese data combined with 50,000 pieces of English data from the open source Stanford-Alpaca.                                                                                                                                                                                                                                                                                                                                             |
| [belle-llama-13b-ext](https://github.com/LianjiaTech/BELLE)            | 链家科技           | 开源 | Extending the vocabulary with additional 50K tokens specific for Chinese and further pretraining these word embeddings on Chinese corpus. Full-parameter finetuning the model with 4M high-quality instruction-following examples.                                                                                                                                                                                                                                                  |
| [Ziya-LLaMA-13B-v1](https://mp.weixin.qq.com/s/IeXgq8blGoeVbpIlAUCAjA) | IDEA研究院        | 开源 | 从LLaMA-13B开始重新构建中文词表，进行千亿token量级的已知的最大规模继续预训练，使模型具备原生中文能力。再经过500万条多任务样本的有监督微调(SFT)和综合人类反馈训练（RM+PPO+HFFT+COHFT+RBRS)，进一步激发和加强各种AI任务能力。                                                                                                                                                                                                                                                                                                                                               |
| [guanaco-7b](https://huggingface.co/JosephusCheung/Guanaco)            | JosephusCheung | 开源 | Guanaco is an advanced instruction-following language model built on Meta's LLaMA 7B model. Expanding upon the initial 52K dataset from the Alpaca model, an additional 534K+ entries have been incorporated, covering English, Simplified Chinese, Traditional Chinese (Taiwan), Traditional Chinese (Hong Kong), Japanese, Deutsch, and various linguistic and grammatical tasks. This wealth of data enables Guanaco to perform exceptionally well in multilingual environments. |
| [phoenix-inst-chat-7b](https://github.com/FreedomIntelligence/LLMZoo)  | 港中文            | 开源 | 基于BLOOMZ-7b1-mt，用Instruction + Conversation数据微调，具体数据见[phoenix-sft-data-v1](https://huggingface.co/datasets/FreedomIntelligence/phoenix-sft-data-v1)                                                                                                                                                                                                                                                                                                                                 |


## 📊 排行榜
### 1、综合能力排行榜
综合能力得分为分类能力、信息抽取能力、阅读理解能力三者得分的平均值。
![lin](pic/total.png)

| 类别	 | 大模型	                    | 总分	   | 排名 |
|-----|-------------------------|-------|----|
| 商用	 | chatgpt-3.5	            | 93.8	 | 1  |
| 开源	 | belle-llama-13b-2m	     | 79.2	 | 2  |
| 商用	 | chatglm官方	              | 76.9	 | 3  |
| 商用	 | 讯飞星火	                   | 76.6	 | 4  |
| 开源	 | belle-llama-13b-ext	    | 71.9	 | 5  |
| 开源	 | phoenix-inst-chat-7b	   | 71.8	 | 6  |
| 开源	 | BELLE-on-Open-Datasets	 | 70.9	 | 7  |
| 开源	 | belle-llama-7b-2m	      | 70.4	 | 8  |
| 开源	 | Ziya-LLaMA-13B-v1	      | 70.2	 | 9  |
| 商用	 | minimax	                | 67.4	 | 10 |
| 开源	 | chatglm-6b	             | 66.1	 | 11 |
| 商用	 | 文心一言	                   | 60.6	 | 12 |
| 开源	 | guanaco-7b	             | 49.9	 | 13 |
| 商用	 | 阿里通义千问	                 | 49.4	 | 14 |


<br><br>
### 2、分类能力排行榜
![lin](pic/classification.png)

| 类别	 | 大模型	                    | 分类能力	 | 排名 |
|-----|-------------------------|-------|----|
| 商用	 | chatgpt-3.5	            | 98	   | 1  |
| 商用	 | chatglm官方	              | 82	   | 2  |
| 开源	 | BELLE-on-Open-Datasets	 | 82	   | 3  |
| 开源	 | belle-llama-13b-2m	     | 82	   | 4  |
| 开源	 | phoenix-inst-chat-7b	   | 82	   | 5  |
| 开源	 | belle-llama-7b-2m	      | 76	   | 6  |
| 开源	 | belle-llama-13b-ext	    | 74	   | 7  |
| 开源	 | Ziya-LLaMA-13B-v1	      | 72	   | 8  |
| 商用	 | 讯飞星火	                   | 70	   | 9  |
| 商用	 | minimax	                | 68	   | 10 |
| 开源	 | chatglm-6b	             | 66	   | 11 |
| 开源	 | guanaco-7b	             | 54	   | 12 |
| 商用	 | 文心一言	                   | 48	   | 13 |
| 商用	 | 阿里通义千问	                 | 44	   | 14 |

<br><br>
### 3、信息抽取能力排行榜
![lin](pic/extract.png)

| 类别	 | 大模型	                    | 信息抽取能力	 | 排名 |
|-----|-------------------------|---------|----|
|商用	|chatgpt-3.5	|88	|1|
|商用	|讯飞星火	|79	|2|
|商用	|chatglm官方	|76	|3|
|开源	|belle-llama-13b-2m	|75	|4|
|商用	|文心一言	|71	|5|
|开源	|chatglm-6b	|69	|6|
|开源	|belle-llama-13b-ext	|65	|7|
|开源	|belle-llama-7b-2m	|64	|8|
|开源	|BELLE-on-Open-Datasets	|62	|9|
|开源	|Ziya-LLaMA-13B-v1	|62	|10|
|开源	|phoenix-inst-chat-7b	|62	|11|
|商用	|minimax	|61	|12|
|商用	|阿里通义千问	|47	|13|
|开源	|guanaco-7b	|45	|14|

<br><br>
### 4、阅读理解能力排行榜
阅读理解能力是一种符合能力，考查针对给定信息的理解能力。
依据给定信息的种类，可以细分为：文章问答、表格问答、对话问答……
![lin](pic/mrc.png)

| 类别	 | 大模型	                    | 阅读理解能力	 | 排名 |
|-----|-------------------------|---------|----|
| 商用	 | chatgpt-3.5	            | 95.3	   | 1  |
| 商用	 | 讯飞星火	                   | 80.7	   | 2  |
| 开源	 | belle-llama-13b-2m	     | 80.7	   | 3  |
| 开源	 | belle-llama-13b-ext	    | 76.7	   | 4  |
| 开源	 | Ziya-LLaMA-13B-v1	      | 76.7	   | 5  |
| 商用	 | minimax	                | 73.3	   | 6  |
| 商用	 | chatglm官方	              | 72.7	   | 7  |
| 开源	 | belle-llama-7b-2m	      | 71.3	   | 8  |
| 开源	 | phoenix-inst-chat-7b	   | 71.3	   | 9  |
| 开源	 | BELLE-on-Open-Datasets	 | 68.7	   | 10 |
| 开源	 | chatglm-6b	             | 63.3	   | 11 |
| 商用	 | 文心一言	                   | 62.7	   | 12 |
| 商用	 | 阿里通义千问	                 | 57.3	   | 13 |
| 开源	 | guanaco-7b	             | 50.7	   | 14 |

<br><br>
#### 4.1 表格问答排行榜（阅读理解细分能力）
表格问答作为阅读理解细分能力，单独列出，但不计入综合能力评分。
专门考查大模型对表格的理解分析能力，常用于数据分析。
![lin](pic/tableQA.png)

| 类别	 | 大模型	                    | 表格问答能力	 | 排名 |
|-----|-------------------------|---------|----|
|商用	|chatgpt-3.5	|93	|1|
|开源	|belle-llama-13b-2m	|75	|2|
|商用	|讯飞星火	|69	|3|
|开源	|belle-llama-13b-ext	|69	|4|
|商用	|chatglm官方	|68	|5|
|开源	|Ziya-LLaMA-13B-v1	|65	|6|
|开源	|chatglm-6b	|59	|7|
|开源	|belle-llama-7b-2m	|59	|8|
|开源	|BELLE-on-Open-Datasets	|48	|9|
|商用	|阿里通义千问	|39	|10|
|商用	|文心一言	|38	|11|

<br><br>
## 🌐 各项能力评分
评分方法：从各个维度给大模型打分，每个维度都对应一个评测数据集，包含若干道题。
每道题依据大模型回复质量给1~5分，将评测集内所有题的得分累加并归一化为100分制，即作为最终得分。

| 类别 | 大模型                    | 分类能力 | 信息抽取能力 | 阅读理解能力 | 综合能力 |
|----|------------------------|------|--------|--------|------|
| 商用 | chatgpt-3.5            | 98   | 88     | 95.3   | 93.8 |
| 商用 | 文心一言                   | 48   | 71     | 62.7   | 60.3 |
| 商用 | chatglm官方              | 82   | 76     | 72.7   | 76.9 |
| 商用 | 讯飞星火                   | 70   | 79     | 80.7   | 76.6 |
| 商用 | 阿里通义千问                 | 44   | 47     | 57.3   | 49.4 |
| 商用 | minimax                | 68   | 61     | 73.3   | 67.4 |
| 开源 | chatglm-6b             | 66   | 69     | 63.3   | 66.1 |
| 开源 | belle-llama-7b-2m      | 76   | 64     | 71.3   | 70.4 |
| 开源 | BELLE-on-Open-Datasets | 82   | 62     | 68.7   | 70.9 |
| 开源 | belle-llama-13b-2m     | 82   | 75     | 80.7   | 79.2 |
| 开源 | belle-llama-13b-ext    | 74   | 65     | 76.7   | 71.9 |
| 开源 | Ziya-LLaMA-13B-v1      | 72   | 62     | 76.7   | 70.2 |
| 开源 | guanaco-7b             | 54   | 45     | 50.7   | 49.9 |
| 开源 | phoenix-inst-chat-7b   | 82   | 62     | 71.3   | 71.8 |

<br><br>
## ⚖️原始评测数据
包含各维度评测集以及大模型输出结果，详见本项目的[eval文件目录](eval)
### 评测样本示例
| # | 分类评测样本                                                                         | 信息抽取评测样本                                                                                                                                                                             | 阅读理解评测样本                                                                                                                                                                                                                                                          |
|---|--------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1 | <div style="width: 250px">请分类以下5种水果：香蕉、西瓜、苹果、草莓、葡萄。</div> | <div style="width: 250px">HR: 你好，我是XYZ公司的招聘主管。我很高兴地通知你，你已经通过了我们的初步筛选，并且我们希望邀请你来参加面试。<br>候选人：非常感谢，我很高兴收到你们的邀请。请问面试的时间和地点是什么时候和哪里呢？<br>HR: 面试的时间是下周二上午10点，地点是我们公司位于市中心的办公室。你会在面试前收到一封详细的面试通知邮件，里面会包含面试官的名字、面试时间和地址等信息。<br>候选人：好的，我会准时出席面试的。请问需要我做哪些准备工作呢？<br>HR: 在面试前，请确保你已经仔细研究了我们公司的业务和文化，并准备好了相关的问题和回答。另外，请务必提前到达面试现场，以便有足够的时间了解我们的公司和环境。<br>候选人：明白了，我会尽最大努力准备好的。非常感谢你的邀请，期待能有机会加入贵公司。<br>HR: 很高兴能和你通话，我们也期待着能和你见面。祝你好运，并期待下周能见到你。<br>基于以上对话，抽取出其中的时间、地点和事件。</div>| <div style="width: 250px">牙医：好的，让我们看看你的牙齿。从你的描述和我们的检查结果来看，你可能有一些牙齦疾病，导致牙齿的神经受到刺激，引起了敏感。此外，这些黑色斑点可能是蛀牙。<br>病人：哦，真的吗？那我该怎么办？<br>牙医：别担心，我们可以为你制定一个治疗计划。我们需要首先治疗牙龈疾病，然后清除蛀牙并填充牙洞。在此过程中，我们将确保您感到舒适，并使用先进的技术和材料来实现最佳效果。<br>病人：好的，谢谢您，医生。那么我什么时候可以开始治疗？<br>牙医：让我们为您安排一个约会。您的治疗将在两天后开始。在此期间，请继续刷牙，使用牙线，并避免吃过于甜腻和酸性的食物和饮料。<br>病人：好的，我会的。再次感谢您，医生。<br>牙医：不用谢，我们会尽最大的努力帮助您恢复健康的牙齿。<br>基于以上对话回答：病人在检查中发现的牙齿问题有哪些？</div> |
| 2 | <div style="width: 250px">将下列单词按词性分类。<br>狗，追，跑，大人，高兴，树  </div>| <div style="width: 250px">给定以下文本段落，提取其中的关键信息。<br>今天早上，纽约市长在新闻发布会上宣布了新的计划，旨在减少治安问题。该计划包括增加派遣警察的人数，以及启动社区倡议，以提高居民对警察工作的支持度。 </div> | <div style="width: 250px">文化艺术报讯 国务院办公厅发布关于2023年部分节假日安排的通知，具体内容如下：元旦：2022年12月31日至2023年1月2日放假调休，共3天。春节：1月21日至27日放假调休，共7天。1月28日（星期六）、1月29日（星期日）上班。清明节：4月5日放假，共1天。劳动节：4月29日至5月3日放假调休，共5天。4月23日（星期日）、5月6日（星期六）上班。端午节：6月22日至24日放假调休，共3天。6月25日（星期日）上班。中秋节、国庆节：9月29日至10月6日放假调休，共8天。10月7日（星期六）、10月8日（星期日）上班。<br>基于以上信息回答：2023年五一假期怎么放假。 </div>|
| 3 | <div style="width: 250px">将下列五个词分为两个组别，每个组别都有一个共同点：狗、猫、鸟、鱼、蛇。 </div>| <div style="width: 250px">在给定的短文中找出三个关键词。<br>西方的哲学历史可上溯至古希腊时期，最重要的哲学流派包括柏拉图学派、亚里士多德学派和斯多葛学派。</div>| <div style="width: 250px">基于以下表格，请问张三的考勤情况<br>员工姓名,日期,上班时间,下班时间,是否迟到,是否早退,是否请假<br>张三,1月1日,8:30,17:30,否,否,否<br>李四,1月1日,9:00,18:00,是,否,否<br>王五,1月1日,8:00,16:30,否,是,否<br>赵六,1月1日,8:30,17:00,否,否,是<br>张三,1月2日,8:00,17:00,否,否,否<br>李四,1月2日,8:30,17:30,否,否,否<br>王五,1月2日,9:00,18:00,是,否,否<br>赵六,1月2日,8:30,17:00,否,否,是</div> |
| 4 | <div style="width: 250px">给定一组文本，将文本分成正面和负面情感。<br>举例文本:<br>这部电影非常出色，值得推荐。我觉得导演做得很好。<br>这场音乐会真是个灾难，我非常失望。</div>| <div style="width: 250px">从以下诗句中提取人物名称：两个黄鹂鸣翠柳，一行白鹭上青天。</div> | <div style="width: 250px">对于给定的问答对，判断问题是否被正确回答<br>问题：地球是第几颗行星？<br>答案：地球是第三颗行星。 </div>  |
| 5 | <div style="width: 250px">将以下10个单词分类为动物或植物。<br>树木、狮子、玫瑰、草地、松鼠、猴子、蘑菇、兔子、山羊、香蕉 | 明天天气怎么样？<br>广州明天最冷多少度？<br>广东大后天最暖多少度？<br>北京冷不冷？<br>提取出上述句子中的地理位置实体</div>  | <div style="width: 250px">根据以下文本回答问题：什么是能量守恒定律？<br>物理学是自然科学的一个重要分支，研究自然界的物质、能量、空间和时间等基本概念。在物理学的研究中，科学家们探索了宇宙的奥秘，从微观粒子到宏观天体的各个方面都进行了深入研究。<br>在物理学中，一个重要的概念是能量守恒定律。能量守恒定律指出，一个孤立系统的总能量在任何时间都是不变的。也就是说，能量不能被创造或摧毁，只能被转化成其他形式。例如，当一个物体被提起并释放时，它的势能被转化为动能，使得物体开始运动。 </div>  |
|……|……|……|……|

### 大模型输出结果示例
|prompt|  <div style="width: 250px">chatgpt3.5</div>  | <div style="width: 250px">百度</div> | …… |
|-|-------------|-------------------------------------------|----|
|<div style="width: 250px">将下列单词按词性分类。<br>狗，追，跑，大人，高兴，树</div>| <div style="width: 250px">好的，以下是这些单词按照词性分类后的结果：<br>名词：狗、大人、树<br>动词：追、跑<br>形容词：高兴 </div>| <div style="width: 250px">动物类别：狗<br>动作类别：追<br>动作类别：跑<br>情感类别：高兴<br>植物类别：树</div>| …… |
|<div style="width: 250px">从以下诗句中提取人物名称：两个黄鹂鸣翠柳，一行白鹭上青天。</div>| <div style="width: 250px">这句诗出自唐代诗人杜甫的《绝句》，并没有明确提到人物名称。这句诗主要是在描绘自然景色，通过形象生动的语言来表现春天的美丽景象。<br>诗中提到“两个黄鹂”，“一行白鹭”，他们是动物而非人物名称。黄鹂是一种鸟类，它们身体鲜黄色，嘴巴呈钩形，叫声清脆悦耳；白鹭是另一种水鸟，身体白色，嘴巴细长，善于捕捉鱼类等。<br>杨婉这句诗意境深远，很好地表现了大自然的美丽和神奇。</div>| <div style="width: 250px">人物名称：黄鹂、白鹭 </div> | …… |
|……|……| ……  |……|