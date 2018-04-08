#encoding=utf-8
import visdom
import pickle
import numpy as np

#把loss值用pickle.dump保存，再pickle.load读取
with open('train_loss_swish.txt', 'rb') as s:
    data_swish = pickle.load(s)

with open('train_loss_relu.txt', 'rb') as r:
    data_relu = pickle.load(r)

with open('train_loss_tanh.txt', 'rb') as t:
    data_tanh = pickle.load(t)

vis = visdom.Visdom()
startup_sec = 1
while not vis.check_connection() and startup_sec > 0:
    time.sleep(0.1)
    startup_sec -= 0.1
assert vis.check_connection(), 'No connection could be formed quickly'

lengh = len(data_swish)

loss_s = []
loss_r = []
loss_t = []
index = []

#400个点取一次平均，让曲线平滑一些
gap = 400

for i in range(1,lengh,gap):
    tmp_r = 0
    tmp_s = 0
    tmp_t = 0
    for m in range(gap):
        tmp_s += data_swish[i]
        tmp_t += data_tanh[i]
        tmp_r += data_relu[i]    
    loss_s.append(tmp_s/gap)
    loss_t.append(tmp_t/gap)
    loss_r.append(tmp_r/gap)
    index.append(i)



vis.line(X=np.array(index),
         Y=np.column_stack((np.array(loss_s), np.array(loss_r), np.array(loss_t))), 
         win='train loss 500', 
         opts=dict(legend=["swish","relu","tanh"],title='train loss'))


s.close()
r.close()
t.close()












