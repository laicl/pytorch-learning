# pytorch-learning

cnn_test.py是基于pytorch写的一个CNN模型。
里面包含了卷积、激活、归一化(normalization)等CNN基本模块。
通过替换基本模块，可以简单测试不同操作对于cnn分类结果的影响。

运行环境：
1. 安装pytorch 
cpu版安装命令
pip3 install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp35-cp35m-linux_x86_64.whl 
pip3 install torchvision

2. 安装visdom(pytorch可视化工具)
pip install visdom  
pip install --upgrade visdom 

运行命令：
1. python -m visdom.server
运行visdom
运行成功后，会提示You can navigate to http://localhost:8097
在浏览器打开http://localhost:8097 或者 直接使用服务器IP, 如http://服务器IP:8097

2. python cnn-test.py (--wn='BN') (--batch_size=32)
(可选)，其他可选参数如下
--wn='vidsom window name' 对应visdom窗口的名字。visdom会根据名字，选择绘制窗口。如果想保留多个窗口，建议使用不同的名字。
--batch_size 对应batch size大小
--epochs 对应epochs值
--lr  对应learning rate学习率

注： 查看conda环境命令为conda info -e