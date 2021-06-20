# 3DCNN团队 方案文档

### 依赖

##### python库：（直接pip最新的就可以）

- ffmpy
- tqdm
- opencv-python
- soundfile
- numpy
- pandas
- torch (>1.6即可)
- torchvision
- timm
- sklearn
- transformers
- scipy

##### 可执行文件：

- ffmpeg (用它的可执行文件的所在路径，替换video2audio.py和video2img.py中executable参数的值。如果是linux已经添加到path中的，那就可以把executable参数去掉)



### 运行方式

1. 将数据集解压，分别为data_A/train、data_A/test、data_B、data_C，其中的必选数据.txt、补充数据.txt移动到和他们同级的目录中
2. video2audio.py 将视频转为音频
3. video2img.py 将视频转为图像
4. audio2melspectrogram.py 将音频转为频谱
5. img2npy.py 将图像转为特征数组
6. prepare.py 将标签转为dataframe形式
7. train_audio.py 训练音频并预测
8. train_img.py 训练图像并预测
9. ensemble.py 将音频和图像的预测融合



### 方案介绍

- 对音频，转换为频谱图后，使用CNN做图像回归，标签为点位除以总长度，为[0,1]区间。A榜最佳能达到3.3，B榜4.3
- 对图像，首先按25FPS抽帧，然后每1s(25帧)的图像经过R(2+1)D模型提取特征，这样每个视频得到(时长/s,特征数)的一个特征向量，可视作一个时序特征，经过GRU后，得到每个位置(秒级)上是片头/片尾的概率，做一个分类任务。A榜最佳能达到3.7，B榜4.2
- 融合时，首先将图像的分类结果转为数值(argmax)，然后简单地和音频的回归结果取平均

