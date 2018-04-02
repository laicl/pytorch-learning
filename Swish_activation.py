#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F

## 由于 Function 可能需要暂存 input tensor。
## 因此，建议不复用 Function 对象，以避免遇到内存提前释放的问题。
class Swish_act(torch.autograd.Function):
    ## save_for_backward can only!!!! save input or output tensors
    @staticmethod
    def forward(self, input_):
        print('swish act op forward')
        output = input_ * F.sigmoid(input_)
        self.save_for_backward(input_)
        return output

    @staticmethod
    def backward(self, grad_output):
	## according to the chain rule(Backpropagation),
	## d(loss)/d(x) = d(loss)/d(output) * d(output)/d(x)
	## grad_output is the d(loss)/d(output)
	## we calculate and save the d(output)/d(x) in forward
        input_, = self.saved_tensors
        output = input_ * F.sigmoid(input_)
        grad_swish = output + F.sigmoid(input_) * (1 - output)
        print('swish act op backward')
        return grad_output * grad_swish
