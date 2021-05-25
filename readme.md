# 手写数字识别器/recognizer for handwritten numbers

## 需求

### 手写数字的识别器

- 功能要求：
  1. 手写数字之后，拍照上传到电脑
  2. 读入照片，能够进行数字分割并显示
  3. 按照顺序读取每一个数字并识别
  4. 打印数字到固定框内
- 其他要求
  - 在Windows上运行
  - 可执行文件双击运行

## 开发思路

### 总体思路

1. python实现手写字的图片分割
2. python实现单个数字被分割之后的灰度处理、分辨率处理（28*28）
3. python实现上述处理之后调用训练好的模型读取并打印
4. python实现GUI框架
5. 打包

### 实现手写数字的图片分割

- 思路
  - 找代码，修改
  - 找到了使用keras的范例：[HandwrittenDigitRecognition](https://github.com/Wangzg123/HandwrittenDigitRecognition)，准备将其修改为pytorch代码

