# SenseVoice ONNX 

基于 [SenseVoiceSmall](https://github.com/FunASR/SenseVoice) 模型实现的高性能、低延迟 ASR 推理引擎。

额外特性：实现**音素级（CTC）的高精度热词召回（Radar）**，支持时间戳、长音频转录，可导出 srt。

### 🚀 核心特性

- ✅ **热词雷达 (Radar)** - 独创的 CTC 框架级热词锚点技术，召回率与准确度双优。
- ✅ **多精度量化** - 可导出 `fp32`, `fp16`, `int4` 量化。int4 体积只有 130MB，仍具有高精度。
- ✅ **长音频转录** - 智能特征分段与合并逻辑，处理数小时音频不卡死。


---

## 📊 性能表现

小新Pro16GT笔记本（RTX5050 + U9-285H）实测：

### GPU (DirectML) 极速模式

使用 fp16 精度，20秒内的音频可在 60ms 内转录完毕。RTF 低于 0.01。

int4 或 int8 精度需要多 20ms，主因是运行时的动态精度转换。

用 DML 时，不同长度的片段推理会有算子重新编译的开销，为了最短的延迟，引擎设置有 pad_to 参数，将过短的音频统一填充到 20s 进行转录，延迟最低。

### CPU 高效模式

10秒钟音频可以 200ms 转录完毕。RTF 约 0.02。


### 显存占用 (VRAM)

除了载入模型需要显存，推理也需要占用显存。实测在 DML 加速时，除了模型所占显存， 20s 的音频需额外占600MB的显存。

---

## 🛠️ 快速上手

### 1. 安装环境


```bash
pip install -r requirements.txt
```

### 2. 准备模型资源

#### 2.1 下载原始模型
从 ModelScope 下载官方 SenseVoiceSmall 模型：

```bash
modelscope download --model iic/SenseVoiceSmall
```

#### 2.2 导出与量化 (Step 01-04)
依次运行以下脚本，将原始模型转换为优化后的 ONNX 格式：

```bash
python 01-Export-Encoder.py    # 导出编码器 (Encoder)
python 02-Export-CTC.py        # 导出解码器 (CTC Decoder)
python 03-Prepare-Assets.py    # 提取 Embedding、Tokenizer 及配置文件
python 04-Quantize-Models.py   # 执行量化 (如导出 int4/int8 版本)
```

---

### 3. 一键推理 (含热词测试)
直接执行 `16` 号脚本即可体验集成了“热词雷达”的推理流程：

```bash
python 16-Final-Hotword-Inference.py
```

```
[SenseVoice] 正在处理音频: test-fun.mp3
[Hotwords] 当前热词列表: hot.txt

[Performance] 开始测速...
 >>> 第 1 轮耗时:   86.63ms | 识别文本: 那我来测试一下Fun-ASR-Nano。
     [细节] Frontend:  0.0ms | Encoder: 70.0ms | Decoder: 16.3ms | Radar:  0.3ms
 >>> 第 2 轮耗时:   84.97ms | 识别文本: 那我来测试一下Fun-ASR-Nano。
     [细节] Frontend:  0.0ms | Encoder: 70.0ms | Decoder: 14.4ms | Radar:  0.2ms
 >>> 第 3 轮耗时:   84.76ms | 识别文本: 那我来测试一下Fun-ASR-Nano。
     [细节] Frontend:  0.0ms | Encoder: 70.0ms | Decoder: 14.5ms | Radar:  0.3ms

==================================================
       时间戳 | 字符         | 类型
--------------------------------------------------
   0.18s | 那          |   Greedy
   0.36s | 我          |   Greedy
   0.48s | 来          |   Greedy
   0.60s | 测          |   Greedy
   0.78s | 试          |   Greedy
   0.96s | 一          |   Greedy
   1.08s | 下          |   Greedy
   1.62s | F          | 🔥 HOTWORD
   1.80s | un-        | 🔥 HOTWORD
   2.10s | AS         | 🔥 HOTWORD
   2.52s | R-         | 🔥 HOTWORD
   2.88s | Na         | 🔥 HOTWORD
   3.06s | no         | 🔥 HOTWORD
   3.42s | 。          |   Greedy
--------------------------------------------------
【检测到的热词】: ['Fun-ASR-Nano']
【最终识别文本】
那我来测试一下Fun-ASR-Nano。
==================================================
```

17号脚本可以直接转录长音频文件，生成 srt 。


---

## 🔥 热词雷达 (Hotword Radar)

项目内置了独具特色的 **“热词雷达”** 组件。它不依赖 Beam Search 等高负荷计算，而是通过以下步骤实现极速召回：

1. **帧驱动搜索**：利用 CTC 后验概率，监控每一帧的非空输出。
2. **Top-K 空间扫描**：即使热词首字没排在第一（Greedy 没出），只要它在 Top-K 的搜索范围内，雷达就能将其捕获。
3. **Case-Insensitive 支持**：通过 Token 映射的小写 Piece 逻辑，支持跨大小写召回（如“Fun-ASR” 对应 “fun-asr”）。

你可以使用 **`15-Debug-CTC-Frames.py`** 直观查看热词是如何在搜索空间中被锚定的：
```text
PS D:\cosyvoice> python .\15-Debug-CTC-Frames.py
[Encoder] 正在初始化 ONNX 会话 (EP: CPUExecutionProvider)...
[Encoder] 自动检测输入精度: <class 'numpy.float32'>
[Decoder] 正在初始化 ONNX 会话 (EP: CPUExecutionProvider)...
[Decoder] 自动检测输入精度: <class 'numpy.float32'>

==================================================
帧时间   | Greedy | 热词锚点  | 搜索空间

--------------------------------------------------
  0.00s    |    .  |       | · 这 嗯 那 我 对 你 好
  0.06s    |    .  |       | · . " ▁ ' ， d >
  0.12s    |    .  |       | · . " ， 。 ' > ▁
  0.18s    | 那    |       | 那 让 · 拿 然 哪 当 但
  0.24s    |    .  |       | · 那 ， 么 后 为 让 我
  0.30s    |    .  |       | · 么 我 ， 那 后 为 个
  0.36s    | 我    |       | 我 么 · 个 不 后 为 会
  0.42s    |    .  |       | · 们 来 我 ， 么 要 就
  0.48s    | 来    |       | 来 · 们 在 再 要 还 來
  0.54s    |    .  |       | · 来 ， 在 要 测 是 的
  0.60s    | 测    |       | 测 设 策 · 试 侧 说 做
  0.66s    |    .  |       | · 测 ， 的 儿 个 ur 一
  0.72s    |    .  |       | · ， 试 的 一 个 。 测
  0.78s    | 试    |       | 试 是 · 实 示 式 数 视
  0.84s    |    .  |       | · 一 ， 试 A 儿 。 的
  0.90s    |    .  |       | · 一 ， 。 的 了 试 你
  0.96s    | 一    |       | 一 1 · 以 你 已 意 的
  1.02s    |    .  |       | · 一 下 个 ， 1 是 这
  1.08s    | 下    |       | 下 · 些 个 项 是 向 一
  1.14s    |    .  |       | · ， 。 啊 儿 的 一 吧
  1.20s    |    .  |       | · ， 。 啊 儿 1 的 A
  1.26s    |    .  |       | · ， 。 、 ？ 啊 的 ,
  1.32s    |    .  |       | · ， 。 ？ 、 是 , 这
  1.38s    |    .  |       | · ， 是 这 它 我 。 他
  1.44s    |    .  |       | · . " ' ， ▁ d >
  1.50s    |    .  |       | · ， 。 s 是 去 不 他
  1.56s    |    .  |       | · s ， 。 - 是 放 F
  1.62s    | 放    | F     | 放 f phone fin · fo F fu
  1.68s    |    .  |       | · e un ， h ho U O
  1.74s    |    .  |       | · un und e n U ， O
  1.80s    |    .  | un    | · und n un ang on umb ong
  1.86s    |    .  |       | · und n N ， ant 。 un
  1.92s    |    .  |       | · ， 。 、 ？ N s n
  1.98s    |    .  |       | · ， 。 、 ？ , s .
  2.04s    |    .  |       | · ， 。 " ' . A ,
  2.10s    | A     | AS    | A as ▁AS · ▁A As S ▁As
  2.16s    |    .  |       | · ， 。 . ？ 、 , A
  2.22s    |    .  |       | · ， 。 , . 、 ？ S
  2.28s    |    .  |       | · ， S , . 。 、 E
  2.34s    | S     |       | S · s ▁S ▁s X E I
  2.40s    |    .  |       | · ， 、 , S 。 . ？
  2.46s    |    .  |       | · 、 , ， 。 . S ▁
  2.52s    | R     | R     | R ▁R r ▁r 二 2 · RA
  2.58s    |    .  |       | · ， 。 、 . , ▁ ？
  2.64s    |    .  |       | · ， 、 。 , ▁ . N
  2.70s    |    .  |       | · ， 。 、 N , . n
  2.76s    |    .  |       | · ， 。 , n N 、 ？
  2.82s    |    .  |       | · n ， N 。 s , -
  2.88s    | n     | Na    | n na no ▁nano · N an ▁no
  2.94s    |    .  |       | · n an un on ur han ar
  3.00s    |    .  |       | · n an un on ll own N
  3.06s    | no    | no    | no · o eno ow nel l ▁no
  3.12s    |    .  |       | · O o l ll U ▁ L
  3.18s    |    .  |       | · O l U L ▁ o ll
  3.24s    |    .  |       | · O Y E e W V l
  3.30s    |    .  |       | · . " ▁ ' > d 。
  3.36s    |    .  |       | · Y . 了 的 啊 y 0
  3.42s    | 。     |       | 。 . · ？ ? ! " ，
==================================================
表格说明：'热词召回结果' 列显示的是雷达在 Top-8 空间中捕捉到的匹配路径。
```

---

## 🏗️ 项目结构

```bash
├── cosyvoice_onnx/           # 核心推理引擎包
│   └── inference/
│       ├── engine.py         # 顶层 Inference 封装
│       ├── encoder.py        # 语音编码器处理逻辑
│       ├── decoder.py        # CTC 解码器处理逻辑
│       ├── radar.py          # 热词雷达高效率匹配算法
│       ├── audio.py          # 音频加载与NumPy 特征提取前端
│       ├── schema.py         # ASR 配置与结果数据结构定义
│       ├── integrator.py     # 识别结果整合逻辑 (Greedy + Hotword)
│       ├── exporters.py      # SRT/TXT/JSON 结果导出工具
│       ├── chinese_itn.py    # 中文数字规范化 (Inverse Text Normalization)
│       └── __init__.py       # 模块导出接口
├── 01-Export-Encoder.py      # [步骤1] 导出 ONNX 编码器
├── 02-Export-CTC.py          # [步骤2] 导出 ONNX 解码器 (CTC)
├── 03-Prepare-Assets.py      # [步骤3] 提取 Embedding 与 Tokenizer
├── 04-Quantize-Models.py     # [步骤4] 模型量化 (Optional)
├── 15-Debug-CTC-Frames.py    # 热词搜索空间可视化调试器
├── 16-Final-Hotword-Inference.py # 生产环境标准的 ASR 推理示例
└── 17-Long-Audio-Transcription.py # 长音频分段转录示例
```

---

## 📜 致谢

- [SenseVoiceSmall](https://github.com/FunASR/SenseVoice) 

---
*本项目专注于极致的本地化推理体验。*
