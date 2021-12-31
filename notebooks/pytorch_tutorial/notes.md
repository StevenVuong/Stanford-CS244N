# Notes:

## Basic Operations
-  PyTorch more flexible than Tensorflow; allowing users to define operations as they go. TF is prefered in industry; but PyTorch as a ML framework for researchers.
-  Tensors are like matrices; but can also represent higher dimensions, e.g. square image with 256 pixels can be `3*256*256` with dimensions RGB
-  `torch.tensor([[0,1][2,3]], dtype=torch.float)`; can initialize with. & Specify tensor type
    -  Or call tensor method: `tensor_variable.float()`
    -  LongTensor are important in NLP as many indices require Long; which is 64bit integer
-  Can initialize from numpy array: `torch.from_numpy(np.array([[1,2],[3,4]]))
-  Or can initialize from another tensor:
    -  `torch.ones_like(old_tensor)`: Tensor of 1s
    -  `torch.zeros_like(old_tensor)`: Tensor of 0s
    -  `torch.rand_like(old_tensor)`: Elements sampled from uniform distribution between 0 and 1
    -  `torch.randn_like(old_tensor)`: Elements sampled from normal dist
-  Can initialize with shape:
    -  `shape=(4,2,2)`; `torch.zeros(shape)`
-  Can use `torch.arange(end)` to get 1d tensor
-  `x.dtype` property to see data type of a tensor; also `.shape` property; or `tensor_variable.size(0)` to see size of a particular dimension
-  Can change shape with `x.view()` method
    -  Can also use `x.reshape()`; but `x.view()` stores contiguously in memory
    -  `x.view()` changes how tensor is read from memory; but not actually stored; so be aware of this
    -  view requires data to be stored contiguously in memory; meaning data is laid out in memory same way elements are read from it
-  Can use `torch.unsqueeze(x, dim)` to add a dimension of size 1 to provided `dim`, where `x` is a tensor
    -  or `x.squeeze()` to get rid of all dimensions with 1 element
    -  `x.numel()` to get total number of elements in a tensor

## Device
Tells PyTorch where to store tensor; where it is determines whether GPU or CPU will behandling computations
-  `x.device` tells us the device of a tensor
-  Can move a tensor from one device to another with `to(device)`

```
if torch.cuda.is_available():
  x.to('cuda') 
```

## Indexing & Operations
-  Can index similar to how we would in numpy
    -  Can also index with tensor similarly
-  Can get a python scalar with `x[idx].item()`
-  Can perform pairwise multiplication; addition; subtraction etc.. Broadcasting happens
-  Can use `tensor.matmul(other_tensor)` for matrix multiplication; or an `@`
-  `x.T` to get transpose; the other dimensions of a tensor may be inferred..
-  Can do `mean(dim)` to get the mean or `std(dim)` to get the standard deviation along a dimension
    -  `x.mean(0)` to get the mean along the 0th dimension
-  Can do `torch.cat` to concatenate tensors
-  Most operations are not in place; but we can by adding an _ at the end of each method name: `a.add_(a)`

## Autograd
-  Can call `backward()` to ask PyTorch to calculate gradients; which are stored in `grad` attribute
    -  `y=x*x*3` (3x^2); `y.backward()`; `x.grad`: dy/dx = d(3x^2)/dx = 6x = 12
    -  If we call for another tensor; `x.grad` updates to be the sum of gradients calculated so far; for backprop; we sum up gradients for a particular neuron before making an update
    -  This is also why we need to run `.zero_grad()` in each rainin iteration; otherwise gradients would keep building up between iterations, causing our updates to be wrong

## Neural Network
-  `import torch.nn as nn`; predefined blocks of torch.nn, and can put these together to create complex networks
### Linear
-  `nn.Linear(H_in, H_out)` to create a linear layer; takes a matrix of `(N, *, H_in)` dimensions and output `(N, *, H_out)`; `*` can be arbitrary number of dims inbetween;
This performs `Ax+b` where A and B are initialized randomly; if we don't want the linear layer to learn bias parameters; we can initialize with `bias=False`
    -  `nn.Linear(H_in, H_out)` to initialize linear layer
    -  `linear_output = linear(input)`; input is a torch tensor. 
    -  can `list(linear.parameters())` to show A and B parameters

Several other preconfigured layers; `nn.Conv2d`, `nn.ConvTranspose2d`, BatchNorm etc.. <br>
Also Activation Functions: `nn.Sigmoid()`, `nn.ReLU()`, `nn.LeakyReLU()`
    -  Activation fns operate on each element separately; so shape of tensors out are same as ones putin

### Putting Together
Can use `nn.Sequential` to pass one output to next input
```
block = nn.Sequential(
    nn.Linear(4, 2),
    nn.Sigmoid()
)

input = torch.ones(2,3,4)
output = block(input)
```

Can build our own by extending the `nn.Module` class; to build on complex modules; and can initialize 
parameters in `__init__` afer calling `__init__` of super class. All attributes we define are treated as parameters; which 
we can learn during training. <br>
Tensors are not parameters; but can be turned into parameters if they are wrapped in `nn.Parameter` class. <br>
All classes extending `nn.module` are expected to implement a `forward(x)` function, where x is a tensor. This is what is called when `forwarda parameter is
passed to our module, such as `model(x)`; see `./nn_module_extension.py`. <br><br>

Then we can call `model(input)` to see what it does; as well as call `list(model.named_parameters())`

## Optimization
Calling `backward()` can calculate gradients; but we also need to know how to update parameters of our models. This is where
optimizers come in. `torch.optim` contains several that we can use; `optim.SGD`, `optim.Adam`; when initializing; we pass
our model parameters, which can be accessed with `model.parameters()`; telling optimizers which values it will be optimizing. <br>
Optimizers also have a `lr` learning rate parameter; and different optimizers have different hyper-parameters.