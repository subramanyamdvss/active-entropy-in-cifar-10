require 'nn'
require 'optim'
require 'image'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'paths'
require 'image'
require 'sys'



dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('CIFAR Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-model', 'convnet', 'type of model to train: convnet | mlp | linear')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 128, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:option('-testIter',1,'no.of iter per testing')
cmd:option('-trainInter',10000,'error vs no.of training examples interval')
cmd:option('-active',true,'applying active learning ')
cmd:text()
opt = cmd:parse(arg)

classes = {'airplane', 'automobile', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
confusion = optim.ConfusionMatrix(classes)


net = nn.Sequential()
if paths.filep(opt.network) then
    net:read(torch.DiskFile(opt.network))
else

net:add(nn.SpatialConvolution(3,16, 5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 2 : filter bank -> squashing -> max pooling
net:add(nn.SpatialConvolution(16,256 ,5, 5))
net:add(nn.ReLU())
net:add(nn.SpatialMaxPooling(2, 2, 2, 2))
      -- stage 3 : standard 2-layer neural network
net:add(nn.Reshape(256*5*5))
net:add(nn.Linear(256*5*5, 128))
net:add(nn.ReLU())
net:add(nn.Linear(128,#classes))
net:add(nn.LogSoftMax())
end

parameters,gradParameters = net:getParameters()
criterion = nn.ClassNLLCriterion()

if (not paths.filep("cifar10torchsmall.zip")) then
    os.execute('wget -c https://s3.amazonaws.com/torch7/data/cifar10torchsmall.zip')
    os.execute('unzip cifar10torchsmall.zip')
end

trainset = torch.load('cifar10-train.t7')
trainset.data  = (trainset.data):double()
trainset.label = trainset.label+1
testset = torch.load('cifar10-test.t7')
testset.label = testset.label +1


ou = image.toDisplayTensor(testset.data[100])
image.save('dvss.png',ou) -- display the 100-th image in dataset
print(classes[testset.label[100]])

-- ignore setmetatable for now, it insert a feature beyond the scope of this tutorial. It sets the index operator.
setmetatable(trainset, 
    {__index = function(t, i) 
                    return {t.data[i], t.label[i]} 
                end}
);


function trainset:size() 
    return self.data:size(1) 
end
trainset.data= (trainset.data):double():cuda()
trainset.label = trainset.label:double():cuda()
print(#(trainset.data)) 
 -- convert the data from a ByteTensor to a DoubleTensor.
print(torch.type(trainset.data))
mean = {} -- store the mean, to normalize the test set in the future
stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainset.data[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainset.data[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

net = net:cuda()
criterion = criterion:cuda()
cudnn.convert(net, cudnn) 
cudnn.fastest = true
lst = trainset
print(#lst.data,#trainset.data)
local trainError = 0
print(trainset:size())
gradParameters:zero()

function entropy(a,net)
  -- print('dvss')
  z = net:forward(a)
  return -0.1*(z:exp()*z)
end

function compare(a,b)
  return entropy(a,net)<entropy(b,net)
end

function subrange(t, first, last)
  local sub = {}
  for i=first,last do
    sub[#sub + 1] = t[i]
  end
  return sub
end

function topk(list, k, comp)
  collectgarbage()
  cutorch.synchronize()
  time = sys.clock()
  local subset = (list.data)
      print(torch.type(subset))
  torch.setdefaulttensortype('torch.CudaTensor' )
  for i = 1,(list:size()),(opt.batchSize*40) do
    xlua.progress(i,list:size())
    local btch = subset:sub(i,math.min(i+opt.batchSize*40-1,list:size()))
    if i == 1 then
      logprob = net:forward(btch)
    else
      logprob = torch.cat({logprob,net:forward(btch)},1)
    end
  end
  cutorch.synchronize()
  print('\ntime taken',sys.clock()-time)
  time = sys.clock()
  collectgarbage()
  subset = -((torch.cmul(logprob,torch.exp(logprob))*(torch.Tensor(#classes,1):fill(1))):t())
  print(torch.type(subset))
  y , list2 = torch.sort(subset,2,true)
  cutorch.synchronize()
  print('time taken',sys.clock()-time)
  time = sys.clock()
  print(torch.type(list2),torch.type(list.data),torch.type(list.label))
  list.data:indexCopy(1,list2[1],(list.data))
  list.label:indexCopy(1,list2[1],(list.label))
  cutorch.synchronize()
  print('time taken',sys.clock()-time)
  time = sys.clock()
  list2:free()
  batch = list.data:sub(1,k)
  labels = list.label:sub(1,k)
  collectgarbage()
  list.data = list.data:sub(k+1,(#(list.data))[1]) 
  collectgarbage()
  list.label = list.label:sub(k+1,(#(list.label))[1]) 
  collectgarbage()
  print('dvss')
  cutorch.synchronize()
  print('time taken',sys.clock()-time)
  time = sys.clock()
  return batch,labels,list;
end 

 
for sz = opt.trainInter , trainset:size(),opt.trainInter do
for j = 1,opt.maxIter do

for t = 1,sz,opt.batchSize do
      -- disp progress
      
      xlua.progress(t, sz)

      list = trainset[{{1,sz}}]
      -- create mini batch
      local inputs = {}
      local targets = {}
      if not active then
        for i = t,math.min(t+opt.batchSize-1,sz) do
         -- load new sample
          local input = trainset.data[i]
          local target = trainset.label[i]
          table.insert(inputs, input)
          table.insert(targets, target)
        end
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f = 0

         -- evaluate function for complete mini batch
         for i = 1,#inputs do
            -- estimate f
            local output = net:forward(inputs[i])
            local err = criterion:forward(output, targets[i])
            f = f + err

            -- estimate df/dW
            local df_do = criterion:backward(output, targets[i])
            net:backward(inputs[i], df_do)

            -- update confusion
            confusion:add(output, targets[i])

            -- visualize?
            if opt.visualize then
               display(inputs[i])
            end
         end
         -- normalize gradients and f(X)
         gradParameters:div(#inputs)
         f = f/#inputs
         trainError = trainError + f

         -- return f and df/dX
         return f,gradParameters
      end
      -- optimize on current mini-batch




      fevalbatch  = function(x)
        -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end
         -- reset gradients
          torch.type(lst.data,lst.label)
          batch,labels , lst = topk(lst,opt.batchSize,compare)

         -- f is the average of all criterions
          local f = 0
         -- evaluate function for complete mini batch
         -- for i = 1,(#batch)[1] do

            -- estimate f
          cutorch.synchronize()
          print('optim time taken',sys.clock()-time)

          time = sys.clock()
          local output = net:forward(batch)
          cutorch.synchronize()
          print('optim time taken',sys.clock()-time)
          time = sys.clock()

          local err = criterion:forward(output, labels)

          f = f + err

            -- estimate df/dW
          cutorch.synchronize()
          print('optim time taken',sys.clock()-time)
          time = sys.clock()
          local df_do = criterion:backward(output, labels)
          net:backward(batch, df_do)

          if opt.visualize then
            display(batch)
          end

          trainError = trainError + f
          cutorch.synchronize()
          print('optim time taken',sys.clock()-time)
          time = sys.clock()
         -- return f and df/dX
          return f,gradParameters
      end

      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
        if not opt.active then
         config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(feval, parameters, config)
       else 
        
        
        config = config or {learningRate = opt.learningRate,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-7}
         optim.sgd(fevalbatch, parameters, config)
         cutorch.synchronize()
         print('optim time taken',sys.clock()-time)
          time = sys.clock()
        
      end
        
      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end
if j%opt.testIter then


  testset.data = testset.data:cuda()   -- convert from Byte tensor to Double tensor

  for i=1,3 do -- over each image channel
    testset.data[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testset.data[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
  end
  
  correct = 0
  for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        correct = correct + 1
    end
  end

  print(correct, 100*correct/10000 .. ' % ')

  class_performance = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
  for i=1,10000 do
    local groundtruth = testset.label[i]
    local prediction = net:forward(testset.data[i])
    local confidences, indices = torch.sort(prediction, true)  -- true means sort in descending order
    if groundtruth == indices[1] then
        class_performance[groundtruth] = class_performance[groundtruth] + 1
    end
  end

  for i=1,#classes do
    print(classes[i], 100*class_performance[i]/1000 .. ' %')
  end
end
end
end



