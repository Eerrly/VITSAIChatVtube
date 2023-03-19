# VITSAIChatVtube
## 说明：
+ **仅个人学习**
+ **基于 [VITS语音在线合成](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai)**
+ **使用VITS语音合成，使用ChatGPT作为AI，使用Live2d作为展示**
+ **只支持Windows平台**
- - -
## python环境
+ [Anaconda](https://www.anaconda.com/) 作为python环境
+ [Python官方地址](https://www.python.org/) 3.9.13
+ [Pip下载地址](https://pypi.python.org/pypi/pip#downloads) 使用 `python setup.py install` 进行安装， 下载依赖库 `pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple` 清华大学镜像
## Model模型
+ `G_953000.pth` 请去 [VITS语音在线合成](https://huggingface.co/spaces/sayashi/vits-uma-genshin-honkai) 下载，放到model文件夹下
## 用法
+ 将openai key写在根目录下的key.txt里，没有自己创建
+ `python main.py`
## 错误解决方法
+ 如果直接跑会报错`No module named 'monotonic_align.core'`。[按照官方的说法]（https://github.com/jaywalnut310/vits），需要先在命令行 `cd` 到 `monotonic_align` 文件夹，然后开始编译，也就是在命令行中输入 `python setup.py build_ext --inplace`