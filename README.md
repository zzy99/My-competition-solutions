2021.6-2021.8

**全球校园AI算法精英大赛——基于多模型迁移预训练文章质量判别**

成绩：目前第一

地址：[2020 DIGIX GLOBAL AI CHALLENGE](https://developer.huawei.com/consumer/cn/activity/devStarAI/algo/competition.html#/preliminary/info/005/introduction)

类型：文本+分类+半监督（+zero shot）

任务：本题目将为选手提供文章数据，参赛选手基于给定的数据构建文章质量判别模型。所提供的数据经过脱敏处理，保证数据安全。基础数据集包含两部分：训练集和测试集。其中训练集给定了该样本的文章质量的相关标签；测试集用于计算参赛选手模型的评分指标，参赛选手需要计算出测试集中每个样本文章质量判断及优质文章的类型。

评估指标：加权平均F1-score

方案：

赛后公布

------

2021.4-2021.6

**BirdCLEF 2021 - Birdcall Identification**

成绩：银牌（48）

地址：[BirdCLEF 2021 - Birdcall Identification](https://www.kaggle.com/c/birdclef-2021)

类型：音频+分类

任务：In this competition, you’ll automate the acoustic identification of birds in soundscape recordings. You'll examine an acoustic dataset to build detectors and classifiers to extract the signals of interest (bird calls). Innovative solutions will be able to do so efficiently and reliably.

评估指标：row-wise micro averaged F1 score

方案：

还是吃了过拟合的亏，错过了一个0.66（B榜十几名）的提交。。

推理代码：https://www.kaggle.com/zzy990106/private-0-66

- 使用7S的随机块进行分类的训练
- 数据增强是关键，是lb0.6到0.7的主要来源，我们的增强有：mixup，random_power，background，white/pink/bandpass noise
- 在train_soundscape上验证，75以前与LB完全同步，到达一定高分后有可能过拟合
- 后处理：滑动窗口，阈值搜索

------

2021.4-2021.5

**第二届“马栏山杯”国际音视频算法大赛——邀请赛——动漫视频片头片尾点位**

成绩：A榜第二，B榜第四

地址：[芒果TV - 算法大赛](https://challenge.ai.mgtv.com/contest/detail/6)

类型：视频+分类/回归

任务：以大赛组织方提供的视频片段数据为基础，提出行之有效且准确的动漫视频点位识别检测方案。参赛选手需要对提供的视频预测出其片头结束的点位和片尾开始的点位。

评估指标：

![img](https://pic3.zhimg.com/v2-a6daef538d4c2befe227d7e2120ff6ee_b.png)

![img](https://pic3.zhimg.com/v2-66cafcfd3055120edd5a1702fff95bd6_b.png)

![img](https://pic1.zhimg.com/v2-5d7d9094a7ba364d49e7330efcbe4b7c_b.png)

m = 1.000 为可忽略误差

方案：

- 片头视频都是200s，片尾视频都是180s，分开做
- 利用ffmpeg提取视频的图像和音频
- 对音频，转换为频谱图后，使用CNN做图像回归，标签为点位除以总长度（[0,1]区间）。A榜3.3，B榜4.3
- 对图像，首先按25FPS抽帧，然后每1s(25帧)的图像经过R(2+1)D模型提取特征，backbone是resnet18，特征维数为512。这样每个视频得到一个尺寸为(时长,特征数)也就是(180/200,512)的特征向量，可视作一个时序特征。经过GRU后，得到每个位置(秒级)上是片头/片尾的概率，做一个分类任务。A榜3.7，B榜4.2
- 融合时，首先将图像的分类结果转为数值(argmax)，然后简单地和音频的回归结果取平均

------

2021.3-2021.5

**2021搜狐校园文本匹配算法大赛**

成绩：初赛第三，复赛第六，决赛第三

地址：[2021 Sohu Campus Document Matching AIgorithm Competition](https://www.biendata.xyz/competition/sohu_2021)

类型：文本+分类

任务：本次比赛的数据均来自人工标注，数据均为文字片段，每两个片段为一组，参赛选手需要为每对文本在两个颗粒度上判断文本对中的两段文字是否匹配。其中，一个颗粒度较为宽泛，两段文字属于一个话题即可视为匹配；另一个颗粒度较为严格，两段文字必须是同一事件才视为匹配。

评估指标：本次评测任务采用macro F1方法，即对A、B两个文件的label分别计算F1值然后求平均，为最终得分。

方案：

其实这个比赛没那么多花里胡哨的，融合就完事了。有些人吹的天花乱坠的技巧，还比不上换个种子，或者阈值调小个0.01。

- 总体思路：以统一性为主要原则。短文本进行填充，长文本进行截断（长文本的信息主要集中在首部），这样统一了短短、短长、长长任务；让labelA和labelB的数据共享同一个模型参数，混合进行多任务训练，彼此协同提升性能。

![img](assets/v2-2f2efeafca159dcd7a021abc265b52db_720w.jpg)

- 交叉验证
- 模型融合：多折融合，多阶段融合，多权重融合
- 对抗训练
- 知识蒸馏：将多个教师模型融合的软标签，提供给学生模型学习，得到一个性能接近多个教师模型的融合，但参数量大大减少的学生模型。最后用20+模型蒸馏出两个base、一个large
- 阈值后处理：由于F1是一个不合理的指标，依赖于0和1的比例，需要将输出的[0,1]区间的概率，按一定阈值划分为0类、1类；对于正负样本比例非常低的B类标签，阈值应该更低（~0.37）

------

2020.12-2021.3

**RANZCR CLiP - Catheter and Line Position Challenge**

成绩：银牌（55）

地址：[RANZCR CLiP - Catheter and Line Position Challenge](https://www.kaggle.com/c/ranzcr-clip-catheter-line-classification/)

类型：图像+分类

任务：In this competition, you’ll detect the presence and position of catheters and lines on chest x-rays. Use machine learning to train and test your model on 40,000 images to categorize a tube that is poorly placed.

评估指标：AUC

方案：

没什么特别的，就是图像分类，没用到分割的注释，比较好用的模型有nfnet_f0、resnet200d、seresnet152d、efficientnet_b5_ns。

前排的秘诀：使用NIH ChestX等外部数据，使用分割注释

------

2020.11-2021.1

**Cassava Leaf Disease Classification**

成绩：铜牌（299）

地址：[Cassava Leaf Disease Classification](https://www.kaggle.com/c/cassava-leaf-disease-classification)

类型：图像+分类

任务：Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage.

评估指标：准确率

方案：

没什么特别的，就是图像分类。全都在卷调参，拉不开差距，以后也不会参加这种比赛了。比较好用的模型有vit_base_patch16_384、efficientnet_b4_ns、resnext50_32x4d。

前排的秘诀：使用这个公开的模型（它其实是在测试集上训练过的）https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2

------

2020.11-2021.1

**华为云“云上先锋”· AI学习赛**

成绩：第四名

地址：[华为云大赛平台](https://competition.huaweicloud.com/information/1000041335/introduction)

类型：图像+分类

任务：本次比赛为AI主题赛中的学习赛。选手可以使用图像分类算法对常见的生活垃圾图片进行分类。

评估指标： 准确率

方案：

没什么特别的，就是图像分类。限定单模，所以用了全部数据训练，当时没用蒸馏可惜了。

------

 2020.8-2020.11

**Lyft Motion Prediction for Autonomous Vehicles**

成绩：银牌（17）

地址：[Lyft Motion Prediction for Autonomous Vehicles](https://www.kaggle.com/c/lyft-motion-prediction-autonomous-vehicles)

类型：图像+回归

任务：In this competition, you’ll apply your data science skills to build motion prediction models for self-driving vehicles. You'll have access to the largest Prediction Dataset ever released to train and test your models. Your knowledge of machine learning will then be required to predict how cars, cyclists,and pedestrians move in the AV's environment.

评估指标： 

![img](https://pic2.zhimg.com/v2-15581377dd539a04ee02919cbc52cdd1_b.gif)

![img](https://pic4.zhimg.com/v2-3cb6e688b72efbffb2afd24e15517f17_b.gif)

方案：

给定汽车前面99帧的图，预测后50帧的轨迹（位置）。乍一看可以用seq2seq，但这里最有效也是最广为使用的方案是将前面的图在通道上按时序拼在一起，然后做一个图像回归。由于涉及到光栅化，训练瓶颈在于CPU，而且图像也太多了，训练都要好几天。按baseline一直训就能有20-的分数了。

------

2020.5-2020.8

**SIIM-ISIC Melanoma Classification**

成绩：银牌（119）

地址：https://www.kaggle.com/c/siim-isic-melanoma-classificatio

类型：图像+分类

任务：In this competition, you’ll identify melanoma in images of skin lesions. In particular, you’ll use images within the same patient and determine which are likely to represent a melanoma. Using patient-level contextual information may help the development of image analysis tools, which could better support clinical dermatologists.

评估指标： AUC

方案：

没什么特别的，就是图像分类。全都在卷调参，拉不开差距，以后也不会参加这种比赛了。这次比赛最大的经验是相信cv而不是lb，如果搬一些lb很高的公共内核，就等着过拟合翻车吧。
