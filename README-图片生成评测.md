# 文生图模型榜单（主打中文场景） v0.3-1106

## 总榜（八个分榜平均）
得分、耗时和花费均为该模型在八个分榜对应数据的算术平均值；按八榜平均得分降序排列。

| 排名 | 大模型 | 总分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2.5-sunburst（low） | 93.02 | 35.4s | 0.041 | 商用 | openAI |
| 2 | gpt-image-2.5-flare（low） | 92.14 | 29.0s | 0.041 | 商用 | openAI |
| 3 | gpt-image-2（medium） | 91.90 | 45.0s | 0.317 | 商用 | openAI |
| 4 | gpt-image-2（high） | 91.45 | 110.2s | 1.249 | 商用 | openAI |
| 5 | gpt-image-2（low） | 89.66 | 23.2s | 0.041 | 商用 | openAI |
| 6 | doubao-seedream-5-0-pro-260628 | 87.29 | 47.0s | 0.300 | 商用 | 豆包 |
| 7 | qwen-image-3.0-pro | 86.55 | 56.2s | 0.250 | 商用 | 阿里巴巴 |
| 8 | qwen-image-3.0 | 86.54 | 67.4s | 0.180 | 商用 | 阿里巴巴 |
| 9 | gemini-3-pro-image-preview（Nano Banana Pro） | 84.16 | 21.2s | 1.072 | 商用 | google |
| 10 | gemini-3.1-flash-image-preview（Nano Banana 2） | 83.79 | 15.7s | 0.640 | 商用 | google |
| 11 | doubao-seedream-5.0-lite | 82.64 | 48.8s | 0.220 | 商用 | 豆包 |
| 12 | qwen-image-2.0-pro | 79.55 | 16.8s | 0.500 | 商用 | 阿里巴巴 |
| 13 | wan2.7-image-pro | 74.58 | 41.3s | 0.500 | 商用 | 阿里巴巴 |
| 14 | wan2.7-image | 72.74 | 20.0s | 0.200 | 商用 | 阿里巴巴 |
| 15 | flux.2-max | 63.05 | 46.4s | 0.500 | 商用 | BlackForestLabs |
| 16 | gemini-2.5-flash-image（Nano Banana） | 60.06 | 13.2s | 0.281 | 商用 | google |
| 17 | glm-image | 55.45 | 82.1s | 0.100 | 开源 | 智谱AI |
| 18 | flux.2-klein-4b | 35.39 | 15.4s | 0.100 | 开源 | BlackForestLabs |

## 1. 单目标计数榜

生成的图片中指定目标物品的数量准确度（如：刚好 3 只鸟）。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2.5-sunburst（low） | 97.5 | 40.8s | 0.044 | 商用 | openAI |
| 2 | gpt-image-2.5-flare（low） | 90.0 | 30.7s | 0.044 | 商用 | openAI |
| 3 | gpt-image-2（medium） | 85.0 | 54.5s | 0.372 | 商用 | openAI |
| 4 | gpt-image-2（high） | 80.0 | 146.3s | 1.478 | 商用 | openAI |
| 5 | qwen-image-2.0-pro | 77.5 | 16.6s | 0.500 | 商用 | 阿里巴巴 |
| 6 | qwen-image-3.0-pro | 75.0 | 50.3s | 0.250 | 商用 | 阿里巴巴 |
| 7 | gpt-image-2（low） | 72.5 | 30.9s | 0.044 | 商用 | openAI |
| 7 | doubao-seedream-5-0-pro-260628 | 72.5 | 35.9s | 0.300 | 商用 | 豆包 |
| 7 | qwen-image-3.0 | 72.5 | 67.6s | 0.180 | 商用 | 阿里巴巴 |
| 7 | doubao-seedream-5.0-lite | 72.5 | 69.9s | 0.220 | 商用 | 豆包 |
| 11 | gemini-2.5-flash-image（Nano Banana） | 60.0 | 14.5s | 0.280 | 商用 | google |
| 12 | gemini-3-pro-image-preview（Nano Banana Pro） | 57.5 | 22.3s | 1.039 | 商用 | google |
| 12 | gemini-3.1-flash-image-preview（Nano Banana 2） | 57.5 | 29.9s | 0.654 | 商用 | google |
| 12 | flux.2-max | 57.5 | 69.5s | 0.500 | 商用 | BlackForestLabs |
| 12 | glm-image | 57.5 | 83.9s | 0.100 | 开源 | 智谱AI |
| 16 | wan2.7-image-pro | 50.0 | 40.0s | 0.500 | 商用 | 阿里巴巴 |
| 17 | wan2.7-image | 47.5 | 24.5s | 0.200 | 商用 | 阿里巴巴 |
| 18 | flux.2-klein-4b | 35.0 | 29.7s | 0.100 | 开源 | BlackForestLabs |

## 2. 负向指令遵循榜

否定提示词执行力，叫它“不要画什么”，它能否完全不画。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2.5-flare（low） | 97.5 | 22.9s | 0.044 | 商用 | openAI |
| 1 | gpt-image-2（low） | 97.5 | 23.0s | 0.044 | 商用 | openAI |
| 1 | qwen-image-3.0 | 97.5 | 56.1s | 0.180 | 商用 | 阿里巴巴 |
| 1 | gpt-image-2（medium） | 97.5 | 49.8s | 0.372 | 商用 | openAI |
| 1 | gpt-image-2（high） | 97.5 | 132.4s | 1.478 | 商用 | openAI |
| 6 | gemini-3-pro-image-preview（Nano Banana Pro） | 95.0 | 20.3s | 1.034 | 商用 | google |
| 7 | gemini-3.1-flash-image-preview（Nano Banana 2） | 91.3 | 14.0s | 0.643 | 商用 | google |
| 8 | doubao-seedream-5.0-lite | 90.0 | 28.7s | 0.220 | 商用 | 豆包 |
| 8 | doubao-seedream-5-0-pro-260628 | 90.0 | 34.8s | 0.300 | 商用 | 豆包 |
| 8 | qwen-image-3.0-pro | 90.0 | 50.8s | 0.250 | 商用 | 阿里巴巴 |
| 11 | gpt-image-2.5-sunburst（low） | 88.8 | 32.4s | 0.044 | 商用 | openAI |
| 12 | qwen-image-2.0-pro | 78.8 | 13.1s | 0.500 | 商用 | 阿里巴巴 |
| 13 | flux.2-max | 73.8 | 34.9s | 0.500 | 商用 | BlackForestLabs |
| 13 | wan2.7-image-pro | 73.8 | 40.3s | 0.500 | 商用 | 阿里巴巴 |
| 15 | gemini-2.5-flash-image（Nano Banana） | 71.3 | 10.9s | 0.280 | 商用 | google |
| 16 | wan2.7-image | 61.3 | 15.9s | 0.200 | 商用 | 阿里巴巴 |
| 17 | glm-image | 60.4 | 76.0s | 0.100 | 开源 | 智谱AI |
| 18 | flux.2-klein-4b | 35.0 | 10.8s | 0.100 | 开源 | BlackForestLabs |

## 3. 空间方位认知榜

检查主体与参照物的左右、上下、前后、接触或排列关系。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2.5-sunburst（low） | 95.0 | 54.6s | 0.044 | 商用 | openAI |
| 2 | gpt-image-2（medium） | 93.8 | 52.4s | 0.365 | 商用 | openAI |
| 2 | gpt-image-2（high） | 93.8 | 137.9s | 1.450 | 商用 | openAI |
| 4 | gpt-image-2.5-flare（low） | 91.3 | 40.0s | 0.044 | 商用 | openAI |
| 5 | doubao-seedream-5.0-lite | 89.6 | 42.1s | 0.220 | 商用 | 豆包 |
| 6 | wan2.7-image | 88.8 | 24.6s | 0.200 | 商用 | 阿里巴巴 |
| 6 | qwen-image-3.0-pro | 88.8 | 57.2s | 0.250 | 商用 | 阿里巴巴 |
| 8 | wan2.7-image-pro | 87.7 | 45.4s | 0.500 | 商用 | 阿里巴巴 |
| 9 | gpt-image-2（low） | 87.5 | 26.2s | 0.044 | 商用 | openAI |
| 9 | doubao-seedream-5-0-pro-260628 | 87.5 | 50.9s | 0.300 | 商用 | 豆包 |
| 9 | qwen-image-3.0 | 87.5 | 62.9s | 0.180 | 商用 | 阿里巴巴 |
| 12 | gemini-3.1-flash-image-preview（Nano Banana 2） | 84.6 | 13.1s | 0.662 | 商用 | google |
| 13 | flux.2-max | 82.5 | 44.1s | 0.500 | 商用 | BlackForestLabs |
| 14 | gemini-2.5-flash-image（Nano Banana） | 79.6 | 12.0s | 0.280 | 商用 | google |
| 15 | gemini-3-pro-image-preview（Nano Banana Pro） | 79.4 | 20.4s | 1.045 | 商用 | google |
| 16 | qwen-image-2.0-pro | 77.0 | 14.4s | 0.500 | 商用 | 阿里巴巴 |
| 17 | flux.2-klein-4b | 71.9 | 16.5s | 0.100 | 开源 | BlackForestLabs |
| 18 | glm-image | 69.6 | 81.3s | 0.100 | 开源 | 智谱AI |

## 4. 中文汉字书写榜

检查指定文字是否写对

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | doubao-seedream-5-0-pro-260628 | 99.5 | 41.4s | 0.300 | 商用 | 豆包 |
| 2 | gpt-image-2（medium） | 98.7 | 52.1s | 0.373 | 商用 | openAI |
| 3 | gpt-image-2（low） | 98.5 | 24.5s | 0.046 | 商用 | openAI |
| 4 | gpt-image-2（high） | 98.2 | 138.1s | 1.479 | 商用 | openAI |
| 5 | qwen-image-3.0-pro | 97.9 | 52.8s | 0.250 | 商用 | 阿里巴巴 |
| 6 | gpt-image-2.5-flare（low） | 94.0 | 54.4s | 0.046 | 商用 | openAI |
| 7 | qwen-image-3.0 | 93.5 | 63.2s | 0.180 | 商用 | 阿里巴巴 |
| 8 | qwen-image-2.0-pro | 93.3 | 14.0s | 0.500 | 商用 | 阿里巴巴 |
| 9 | gemini-3-pro-image-preview（Nano Banana Pro） | 92.5 | 22.5s | 1.051 | 商用 | google |
| 10 | wan2.7-image | 92.1 | 14.9s | 0.200 | 商用 | 阿里巴巴 |
| 11 | wan2.7-image-pro | 89.9 | 38.5s | 0.500 | 商用 | 阿里巴巴 |
| 12 | gpt-image-2.5-sunburst（low） | 89.1 | 62.0s | 0.046 | 商用 | openAI |
| 13 | gemini-3.1-flash-image-preview（Nano Banana 2） | 87.9 | 14.9s | 0.642 | 商用 | google |
| 14 | doubao-seedream-5.0-lite | 86.7 | 36.9s | 0.220 | 商用 | 豆包 |
| 15 | glm-image | 70.0 | 79.5s | 0.100 | 开源 | 智谱AI |
| 16 | flux.2-max | 47.8 | 42.4s | 0.500 | 商用 | BlackForestLabs |
| 17 | gemini-2.5-flash-image（Nano Banana） | 15.2 | 10.7s | 0.280 | 商用 | google |
| 18 | flux.2-klein-4b | 1.3 | 14.4s | 0.100 | 开源 | BlackForestLabs |


## 5. 多类别分别计数榜

检查同一张图片中多个类别的物品是否分别达到指定数量，避免漏数、混数或类别错配。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2.5-flare（low） | 96.7 | 24.5s | 0.035 | 商用 | openAI |
| 2 | gpt-image-2（medium） | 96.4 | 46.2s | 0.256 | 商用 | openAI |
| 3 | gpt-image-2.5-sunburst（low） | 96.3 | 28.9s | 0.035 | 商用 | openAI |
| 4 | gpt-image-2（low） | 95.0 | 27.3s | 0.035 | 商用 | openAI |
| 5 | gemini-3.1-flash-image-preview（Nano Banana 2） | 93.2 | 15.8s | 0.663 | 商用 | google |
| 6 | gpt-image-2（high） | 92.0 | 100.4s | 1.002 | 商用 | openAI |
| 7 | doubao-seedream-5-0-pro-260628 | 90.1 | 62.9s | 0.300 | 商用 | 豆包 |
| 8 | qwen-image-3.0-pro | 88.6 | 64.1s | 0.250 | 商用 | 阿里巴巴 |
| 9 | qwen-image-2.0-pro | 88.5 | 16.2s | 0.500 | 商用 | 阿里巴巴 |
| 10 | gemini-3-pro-image-preview（Nano Banana Pro） | 87.6 | 21.8s | 1.063 | 商用 | google |
| 11 | doubao-seedream-5.0-lite | 87.6 | 65.0s | 0.220 | 商用 | 豆包 |
| 12 | qwen-image-3.0 | 87.0 | 66.3s | 0.180 | 商用 | 阿里巴巴 |
| 13 | wan2.7-image-pro | 76.8 | 43.5s | 0.500 | 商用 | 阿里巴巴 |
| 14 | wan2.7-image | 68.0 | 21.3s | 0.200 | 商用 | 阿里巴巴 |
| 15 | gemini-2.5-flash-image（Nano Banana） | 68.0 | 17.5s | 0.280 | 商用 | google |
| 16 | flux.2-max | 63.1 | 64.7s | 0.500 | 商用 | BlackForestLabs |
| 17 | glm-image | 42.7 | 84.4s | 0.100 | 开源 | 智谱AI |
| 18 | flux.2-klein-4b | 35.6 | 28.6s | 0.100 | 开源 | BlackForestLabs |

## 6. 数值→矩形面积榜

检查矩形面积是否准确表达指定数值的比例或占比。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2.5-sunburst（low） | 80.0 | 18.2s | 0.037 | 商用 | openAI |
| 2 | gpt-image-2.5-flare（low） | 71.3 | 24.7s | 0.037 | 商用 | openAI |
| 2 | gemini-3-pro-image-preview（Nano Banana Pro） | 71.3 | 20.7s | 1.155 | 商用 | google |
| 2 | gpt-image-2（high） | 71.3 | 64.6s | 0.994 | 商用 | openAI |
| 5 | gemini-3.1-flash-image-preview（Nano Banana 2） | 67.5 | 12.5s | 0.631 | 商用 | google |
| 5 | gpt-image-2（low） | 67.5 | 17.5s | 0.037 | 商用 | openAI |
| 7 | gpt-image-2（medium） | 66.3 | 30.2s | 0.255 | 商用 | openAI |
| 8 | qwen-image-3.0 | 63.8 | 104.9s | 0.180 | 商用 | 阿里巴巴 |
| 9 | doubao-seedream-5-0-pro-260628 | 62.5 | 54.9s | 0.300 | 商用 | 豆包 |
| 9 | qwen-image-3.0-pro | 62.5 | 67.4s | 0.250 | 商用 | 阿里巴巴 |
| 11 | wan2.7-image | 60.0 | 13.1s | 0.200 | 商用 | 阿里巴巴 |
| 12 | doubao-seedream-5.0-lite | 58.8 | 61.9s | 0.220 | 商用 | 豆包 |
| 13 | wan2.7-image-pro | 57.5 | 46.6s | 0.500 | 商用 | 阿里巴巴 |
| 14 | qwen-image-2.0-pro | 48.8 | 15.8s | 0.500 | 商用 | 阿里巴巴 |
| 15 | flux.2-max | 41.3 | 36.2s | 0.500 | 商用 | BlackForestLabs |
| 16 | gemini-2.5-flash-image（Nano Banana） | 38.8 | 15.4s | 0.287 | 商用 | google |
| 17 | glm-image | 27.5 | 78.6s | 0.100 | 开源 | 智谱AI |
| 18 | flux.2-klein-4b | 21.3 | 7.4s | 0.100 | 开源 | BlackForestLabs |

## 7. 字体类别榜

检查指定文字是否使用要求的字体类别，如黑体、宋体、楷体或手写体。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gemini-3-pro-image-preview（Nano Banana Pro） | 99.2 | 20.1s | 1.117 | 商用 | google |
| 2 | gpt-image-2（low） | 98.8 | 19.6s | 0.039 | 商用 | openAI |
| 2 | qwen-image-3.0 | 98.8 | 59.9s | 0.180 | 商用 | 阿里巴巴 |
| 2 | gpt-image-2（high） | 98.8 | 81.1s | 1.087 | 商用 | openAI |
| 5 | doubao-seedream-5-0-pro-260628 | 97.9 | 49.6s | 0.300 | 商用 | 豆包 |
| 6 | gpt-image-2.5-sunburst（low） | 97.5 | 28.2s | 0.039 | 商用 | openAI |
| 6 | gpt-image-2（medium） | 97.5 | 35.6s | 0.279 | 商用 | openAI |
| 8 | gpt-image-2.5-flare（low） | 96.3 | 17.1s | 0.039 | 商用 | openAI |
| 9 | qwen-image-3.0-pro | 95.0 | 51.6s | 0.250 | 商用 | 阿里巴巴 |
| 10 | gemini-3.1-flash-image-preview（Nano Banana 2） | 92.5 | 12.3s | 0.617 | 商用 | google |
| 11 | doubao-seedream-5.0-lite | 87.1 | 42.9s | 0.220 | 商用 | 豆包 |
| 12 | wan2.7-image-pro | 81.7 | 37.4s | 0.500 | 商用 | 阿里巴巴 |
| 13 | qwen-image-2.0-pro | 79.6 | 26.3s | 0.500 | 商用 | 阿里巴巴 |
| 14 | wan2.7-image | 78.8 | 14.2s | 0.200 | 商用 | 阿里巴巴 |
| 15 | gemini-2.5-flash-image（Nano Banana） | 73.8 | 12.7s | 0.280 | 商用 | google |
| 16 | flux.2-max | 64.2 | 42.7s | 0.500 | 商用 | BlackForestLabs |
| 17 | glm-image | 61.3 | 89.8s | 0.100 | 开源 | 智谱AI |
| 18 | flux.2-klein-4b | 38.8 | 8.6s | 0.100 | 开源 | BlackForestLabs |

## 8. 文字修饰榜

检查描边、阴影、下划线、删除线或高亮等文字修饰是否准确作用于指定文字。

| 排名 | 大模型 | 专项得分 | 平均耗时 | 平均花费（元） | 类别 | 机构 |
|---:|---|---:|---:|---:|---|---|
| 1 | gpt-image-2（low） | 100.0 | 16.9s | 0.037 | 商用 | openAI |
| 1 | gpt-image-2.5-flare（low） | 100.0 | 17.3s | 0.037 | 商用 | openAI |
| 1 | gpt-image-2.5-sunburst（low） | 100.0 | 18.0s | 0.037 | 商用 | openAI |
| 1 | gpt-image-2（medium） | 100.0 | 39.0s | 0.263 | 商用 | openAI |
| 1 | gpt-image-2（high） | 100.0 | 80.9s | 1.025 | 商用 | openAI |
| 6 | doubao-seedream-5-0-pro-260628 | 98.3 | 45.8s | 0.300 | 商用 | 豆包 |
| 7 | gemini-3.1-flash-image-preview（Nano Banana 2） | 95.8 | 12.9s | 0.607 | 商用 | google |
| 8 | qwen-image-3.0-pro | 94.6 | 55.1s | 0.250 | 商用 | 阿里巴巴 |
| 9 | qwen-image-2.0-pro | 92.9 | 18.0s | 0.500 | 商用 | 阿里巴巴 |
| 10 | qwen-image-3.0 | 91.7 | 58.0s | 0.180 | 商用 | 阿里巴巴 |
| 11 | gemini-3-pro-image-preview（Nano Banana Pro） | 90.8 | 21.5s | 1.076 | 商用 | google |
| 12 | doubao-seedream-5.0-lite | 88.8 | 43.0s | 0.220 | 商用 | 豆包 |
| 13 | wan2.7-image | 85.4 | 31.3s | 0.200 | 商用 | 阿里巴巴 |
| 14 | wan2.7-image-pro | 79.2 | 38.5s | 0.500 | 商用 | 阿里巴巴 |
| 15 | flux.2-max | 74.2 | 37.1s | 0.500 | 商用 | BlackForestLabs |
| 16 | gemini-2.5-flash-image（Nano Banana） | 73.8 | 12.2s | 0.280 | 商用 | google |
| 17 | glm-image | 54.6 | 83.1s | 0.100 | 开源 | 智谱AI |
| 18 | flux.2-klein-4b | 44.2 | 7.1s | 0.100 | 开源 | BlackForestLabs |