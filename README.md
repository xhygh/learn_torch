# learn_torch

```
导入torch7写的t7 sequential模型，并保存该模型参数
```
## torh7导入模型
按照官网指导安装torch之后，

导入模型失败如下：
![1.png](https://github.com/xhygh/learn_torch/blob/master/torch7%E5%AD%A6%E4%B9%A0/1.png?raw=true)

添加require 'nn' 后成功导入
```
print(net)  # 可以打印出模型的结构，非常方便
```


## torch7导出参数

# 没那麼简单 就能找到 能用的func~ #
net.getParameters():得到所有参数 一！维！数！组！，没错是一维数组，这个函数说明也是flatten。。。
找了一天，放弃了torch


## pytorch导入模型

t7格式的文件需要如下方法导入（pycharm，pytorch）：
```python
from torch.utils.serialization import load_lua
net = load_lua('/home/h/PycharmProjects/chair.t7')
```

## pytorch导出参数

以为在python里面就方便了，想多拉~~~

查到pytorch有一键导出参数函数：

```python
# 保存和加载整个模型
torch.save(model_object, 'model.pkl')
model = torch.load('model.pkl')
# 仅保存和加载模型参数
torch.save(model_object.state_dict(), 'params.pkl')
model_object.load_state_dict(torch.load('params.pkl'))

http://blog.csdn.net/u012436149/article/details/68948816
```

然而，报错说，sequential模型没有state_dict（）属性。。。。

使用debug模式，可以发现
```python
net.modules  #有每层参数，打算手动拿出来
```

写了个函数手动提取参数：
```python
import torch
import torchvision.models as models
from torch.utils.serialization import load_lua
import h5py
import numpy as np

conv = [0,3,6,9,12]
bn = [1,4,7,10]
re = [2,5,8,11]
sig = [13]

net = load_lua('/home/h/PycharmProjects/chair.t7')
b= net.modules


f = open('Gnet.txt','w')
f5 = h5py.File('Gparams.hdf5','w')
for i in range(14):
    if i in conv:
        grp = f5.create_group(name='/'+str(i)+str(b[i]))  #创建本层的grp
        bias = grp.create_dataset(name='bias'+str(i), data=b[i].bias.numpy())  #存入本层bias
        weight = grp.create_dataset(name='weight' + str(i), data=b[i].weight.numpy())  #存入本层weight
        f.write(str(i)+str(b[i])+'\n')
        f.write('bias.shape=%s,weight.shape=%s\n'%(np.array(b[i].bias.size()),
                                                   np.array(b[i].weight.size())))
        f.write('(dH,dT,dW):(%g,%g,%g);(kH,kT,kW):(%g,%g,%g)(padH,padT,padW):(%g,%g,%g)\n'%(b[i].dH, b[i].dT, b[i].dW,
                                                                                          b[i].kH, b[i].kT, b[i].kW,
                                                                                          b[i].padH, b[i].padT, b[i].padW))
        f.write('*******************************************************************************\n')
    if i in bn:
        grp = f5.create_group(name='/' + str(i) + str(b[i]))  # 创建本层的grp
        bias = grp.create_dataset(name='bias' + str(i), data=b[i].bias.numpy())  # 存入本层bias
        weight = grp.create_dataset(name='weight' + str(i), data=b[i].weight.numpy())  # 存入本层weight
        running_mean = grp.create_dataset(name='running_mean' + str(i), data=b[i].running_mean.numpy())  
        running_var = grp.create_dataset(name='running_var' + str(i), data=b[i].running_var.numpy())  
        save_mean = grp.create_dataset(name='save_mean' + str(i), data=b[i].save_mean.numpy())  
        save_std = grp.create_dataset(name='save_std' + str(i), data=b[i].save_std.numpy()) 
        f.write(str(i) + str(b[i]) + '\n')
        f.write('eps:%g,momentum:%g,nDim%g,affine:%s'%(b[i].eps, b[i].momentum, b[i].nDim, str(b[i].affine)))
        f.write('bias.shape=%s,weight.shape=%s\n' % (np.array(b[i].bias.size()),
                                                     np.array(b[i].weight.size())))
        f.write('running_mean.shape=%s,running_var.shape=%s\n' % (np.array(b[i].running_mean.size()),
                                                                 np.array(b[i].running_var.size())))
        f.write('save_mean.shape=%s,save_std.shape=%s\n' % (np.array(b[i].save_mean.size()),
                                                            np.array(b[i].save_std.size())))
        f.write('*******************************************************************************\n')
    if i in re:
        f.write(str(i) + str(b[i]) + '\n')
        f.write('*******************************************************************************\n')
    if i in sig:
        f.write(str(i) + str(b[i]) + '\n')
        f.write('*******************************************************************************\n')


f.close()
f5.close()
```

先凑合用着，发现问题再说吧。

其实TORCH挺好用的，尤其是立即看到结果这一功能太棒了，TF还得创建session，很多东西看不到，超级烦人，然而FB不能再给力点吗，pytorch好歹是自家出的，对自家的兼容性好点呗。
