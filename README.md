# Thisoe's Temp Communicate Board

## 24年6月
### 试用[CNN图像检索](https://github.com/filipradenovic/cnnimageretrieval-pytorch/tree/master)repo
#### \#步骤
0. 想用git的话[安装git](https://gitforwindows.org/)
1. 在“开始”以管理员身份打开cmd，安装`PyTorch`和`torchvision`
   ```bat
   pip install torchvision
   pip3 install torch torchvision torchaudio
   ```
   > 用`pip list`检查安装结果。若安装成功，列表中应包括：
   > ```
   > torch             2.3.0
   > torchaudio        2.3.0
   > torchvision       0.18.0
   > ```
2. 从该repo的[v1.2版发布页面](https://github.com/filipradenovic/cnnimageretrieval-pytorch/releases/tag/v1.2)下载源代码压缩包（[点此直接下载](https://github.com/filipradenovic/cnnimageretrieval-pytorch/archive/refs/tags/v1.2.zip)）
3. 新建空文件夹（下称“根目录”），解压后的三个东西搬进去
4. 根目录创建新记事本，改名`setup.py`，右键用记事本编辑，把[这些玩意](https://github.com/filipradenovic/cnnimageretrieval-pytorch/blob/master/setup.py)粘贴进去，然后`Ctrl`+`H`在第一栏粘贴
   ```
   >=1.0.0,<1.4.0
   ```
   第二栏留空，点`全部替换`。
5. 根目录开cmd，依次执行：
   ```bat
   git init
   git add -A
   git commit -am "init"
   cd cirtorch
   py examples\test.py -npath networks\imageretrievalnet
   ```
6. 




## 24年5月
<details><summary><b>实时</b></summary><br/>
[油管教程](https://www.youtube.com/watch?v=2S1dgHpqCdk&list=PLhhyoLH6IjfxeoooqP9rhU3HJIAVAJ3Vz&index=1)

[pip 安装程序源](https://bootstrap.pypa.io/get-pip.py)
(Ctrl+S保存文件，在文件管理器的地址栏写cmd，使用指令`py get-pip.py`)
<hr>
</details>

<details><summary><b>准备</b></summary><br/>
1. 下载 [Anaconda](https://www.anaconda.com/download/success)
2. 下载 [PyCharm IDE](https://www.jetbrains.com/pycharm/download/?section=windows)
3. [Python 官网](https://www.python.org/downloads/)安装 Python （PowerShell检查：`py --version`）
4. 安装 pip （PowerShell检查：`pip3 -V`）
5. PowerShell 安装 PyTorch: `pip3 install torch torchvision torchaudio` （）
6. 后台开个小抄 [CONDA CHEAT SHEET](https://docs.conda.io/projects/conda/en/4.6.0/_downloads/52a95608c49671267e40c689e0bc00ca/conda-cheatsheet.pdf)
<hr>
</details>

<details><summary><b>用 Simple Transformers 包</b></summary><br/>
> [教程](https://youtu.be/u--UVvH-LIQ)
> 
> GitHub: [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers)

1. 安装 [simpletransformers](https://simpletransformers.ai/docs/installation/)

https://simpletransformers.ai/docs/installation/
<hr>
</details>
