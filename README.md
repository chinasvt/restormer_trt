## 总述
应付完成比赛，可以不用评估了:joy:，包括：
- 图像去模糊 Restormer(https://github.com/swz30/Restormer.git)
- 工作就是转了下模型跑了速度和数值比较，其余优化实在没时间优化
- 在Docker里面代码编译、运行步骤的完整说明
  - 编译 cd /root/workspace/restormer_trt && mkdir build && cd build && cmake ../
  - 运行 ./demo
  - 测试 cd /root/workspace/restormer_trt/tools && python3 test_performance.py

## 原始模型
### 模型简介
请介绍模型的基本信息，可以包含但不限于以下内容：
- 去除图像模糊or噪声
- 业界使用情况着实不清楚，纯粹为了比赛选的
- 模型也是基于transformer，具体情况不清楚
  ![img.png](img.png)

### 模型优化的难点
- 目前遇到模型导出onnx，PixelShuffle和PixelUnShuffle不支持问题

## 优化过程
属实应付，不用评估了

## 精度与加速效果
| Framework    | Resolution | Precision | Elapsed Time |
| ------------ | ---------- | --------- | ------------ |
| Pytorch      |   368x552  |    FP32   |   296.434ms  |
| TensorRT     |   368x552  |    FP16   |   192.337ms  |

## 经验与体会（可选）
属实没精力，应付一波完成比赛
