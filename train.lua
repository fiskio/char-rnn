--[[

This file trains a character-level multi-layer RNN on text data

Code is based on implementation in
https://github.com/oxford-cs-ml-2015/practical6
but modified to have multi-layer support, GPU support, as well as
many other common model/optimization bells and whistles.
The practical6 code is in turn based on
https://github.com/wojciechz/learning_to_execute
which is turn based on other stuff in Torch, etc... (long lineage)

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'
require 'autobw'

require 'util.OneHot'
require 'util.misc'
require 'util.HLogSoftMax'
require 'util.Squeeze'
local text = require 'text'
local model_utils = require 'util.model_utils'
HSMClass = require 'util.HSMClass'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'
local SCRNN = require 'model.SCRNN'


cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a recurrent language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/ptb','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-hidden_size', 512, 'size of LSTM internal state')
cmd:option('-context_size', 128, 'size of SCRNN context state')
cmd:option('-num_layers', 1, 'number of layers in the LSTM')
cmd:option('-model', 'rnn', 'rnn | gru | lstm | scrnn')
cmd:option('-embeddings', 128, 'size of word embeddings')
cmd:option('-hsm', -1, 'HSM classes, 0 is off, -1 is sqrt(vocab)')
-- optimization
cmd:option('-optim', 'rmsprop', 'Optimisation algorithm')
cmd:option('-learning_rate',1e-3,'learning rate')
cmd:option('-learning_rate_decay',0.95,'learning rate decay')
cmd:option('-learning_rate_decay_after',25,'in number of epochs, when to start decaying the learning rate')
cmd:option('-dropout',0,'dropout for regularization, used after each RNN hidden layer. 0 = no dropout')
cmd:option('-sgd_weight_decay', 0, 'weight decay')
cmd:option('-sgd_momentum', 0, 'momentum')
cmd:option('-sgd_momentum_nesterov', false, 'use nesterov momentum ')
cmd:option('-rmsprop_alpha', 0.99,'smoothing constant')
cmd:option('-rmsprop_epsilon', 1e-8, 'value with which to inistialise m')
cmd:option('-adam_beta1', 0.9, 'first moment coefficient')
cmd:option('-adam_beta2', 0.999, 'second moment coefficient')
cmd:option('-adam_lambda', 1-1e-8, 'first moment decay')
cmd:option('-adadelta_rho', 0.95, 'interpolation parameter')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',32,'number of sequences to train on in parallel')
cmd:option('-max_epochs',50,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at this value')
cmd:option('-init_from', '', 'initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed',42,'torch manual random number generator seed')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile','lstm','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('package cunn not found!') end
    if not ok2 then print('package cutorch not found!') end
    if ok and ok2 then
        print('using CUDA on GPU ' .. opt.gpuid .. '...')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
        print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- initialize clnn/cltorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        print('using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the data loader class
local loader = Text{
                  name = paths.basename(opt.data_dir),
                  data_path = paths.dirname(opt.data_dir),
                  vocab_size = 1e4,
                  batch_size = opt.batch_size,
                  max_length = 50,
                  max_reps = 10
               }
local vocab_size = loader._vocab_size  -- the number of distinct characters
local vocab = loader._word2class
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
if not path.exists(opt.checkpoint_dir) then lfs.mkdir(opt.checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c,i in pairs(checkpoint.vocab) do
        if not vocab[c] == i then
            vocab_compatible = false
        end
    end
    assert(vocab_compatible, 'error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('overwriting hidden_size=' .. checkpoint.opt.hidden_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.hidden_size = checkpoint.opt.hidden_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else
    print('creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.hidden_size, opt.num_layers, opt.embeddings, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.hidden_size, opt.num_layers, opt.embeddings, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.hidden_size, opt.num_layers, opt.embeddings, opt.dropout, opt.hsm)
    elseif opt.model == 'scrnn' then
        protos.rnn = SCRNN.scrnn(vocab_size, opt.embeddings, opt.hidden_size, opt.context_size, opt.num_layers, opt.dropout)
    end
    -- HSM?
    if opt.hsm ~= 0 then
       print(loader:hsm_mapping():size())
       protos.criterion = nn.HLogSoftMax(loader:hsm_mapping(), opt.hidden_size)
    else
       protos.criterion = nn.ClassNLLCriterion()
    end
end

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    if opt.model == 'scrnn' then
       local s_init = torch.zeros(opt.batch_size, opt.context_size)
       if opt.gpuid >=0 and opt.opencl == 0 then s_init = s_init:cuda() end
       if opt.gpuid >=0 and opt.opencl == 1 then s_init = s_init:cl() end
       table.insert(init_state, s_init:clone())
    end
    local h_init = torch.zeros(opt.batch_size, opt.hidden_size)
    if opt.gpuid >=0 and opt.opencl == 0 then h_init = h_init:cuda() end
    if opt.gpuid >=0 and opt.opencl == 1 then h_init = h_init:cl() end
    table.insert(init_state, h_init:clone())
    if opt.model == 'lstm' then
        table.insert(init_state, h_init:clone())
    end
end

-- ship the model to the GPU if desired
if opt.gpuid >= 0 and opt.opencl == 0 then
    for k,v in pairs(protos) do v:cuda() end
end
if opt.gpuid >= 0 and opt.opencl == 1 then
    for k,v in pairs(protos) do v:cl() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.rnn, protos.criterion)

-- initialization
if do_random_init then
   -- TODO improve initialisation
   params:uniform(-0.08, 0.08) -- small numbers uniform
end

print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    -- seq_length + 1 because Text prepends start of sequence token
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length+1, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    --print('evaluating loss over split index ' .. split_index)
    local n = #loader._valid_batches
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(loader:valid_batches()) -- move batch iteration pointer for this split to front
    local loss = 0
    xlua.progress(1,n)
    --local rnn_state

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(loader:valid_batches())
        local init_state_local = {}
        for i,t in ipairs(init_state) do
           init_state_local[i] = torch.zeros(x:size(1), t:size(2))
        end
         if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
            for i,s in ipairs(init_state_local) do init_state_local[i] = s:cuda() end
        end
        if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
            x = x:cl()
            y = y:cl()
            for i,s in ipairs(init_state_local) do init_state_local[i] = s:cl() end
        end
       local rnn_state = {[0] = init_state_local}
        -- forward pass
        local curr_loss = 0
        for t=1, x:size(2) do
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
            rnn_state[t] = {}
            for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end
            curr_loss = curr_loss + clones.criterion[t]:forward(lst[#lst], y[{{}, t}])
        end
        -- carry over lstm state
        loss = loss + curr_loss / x:size(2)
        xlua.progress(i, n)
        collectgarbage()
    end

    xlua.progress(n, n)
    loss = loss / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()
    --if opt.hsm ~= 0 then hsm_grad_params:zero() end

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(loader._train_batches)
    local init_state_local = {}
    for i,t in ipairs(init_state) do
          init_state_local[i] = torch.zeros(x:size(1), t:size(2))
    end

    if opt.gpuid >= 0 and opt.opencl == 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
        for i,s in ipairs(init_state_local) do init_state_local[i] = s:cuda() end
    end
    if opt.gpuid >= 0 and opt.opencl == 1 then -- ship the input arrays to GPU
        x = x:cl()
        y = y:cl()
        for i,s in ipairs(init_state_local) do init_state_local[i] = s:cuda() end
    end

    local tape = autobw.Tape()

    ------------------- forward pass -------------------
    tape:begin()

    local rnn_state = {[0] = init_state_local}
    local predictions = {}
    local loss = 0
    for t=1,x:size(2) do
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        local lst = clones.rnn[t]:forward{x[{{}, t}], unpack(rnn_state[t-1])}
        rnn_state[t] = {}
        for i=1,#init_state do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
        loss = loss + clones.criterion[t]:forward(lst[#lst], y[{{}, t}])
    end
    loss = loss / x:size(2)

    tape:stop()
    ------------------ backward pass -------------------
    tape:backward()
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    -- init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    -- TODO renorm?
    return loss, grad_params
end

-- start optimization here
train_losses = {}
val_losses = {}

local optim_states = {
   rmsprop  = { learningRate = opt.learning_rate,
                alpha        = opt.rmsprop_alpha,
                epsilon      = opt.rmsprop_epsilon },
   adagrad  = { learningRate      = opt.learning_rate,
                learningRateDecay = opt.learning_rate_decay },
   adam     = { beta1  = opt.adam_beta1,
                beta2  = opt.adam_beta2,
                lambda = opt.adam_lambda },
   adadelta = { rho = opt.adadelta_rho },
   sgd      = { learningRate      = opt.learning_rate,
                learningRateDecay = opt.learning_rate_decay,
                weightDecay       = opt.sgd_weight_decay,
                momentum          = opt.sgd_momentum,
                nesterov          = opt.sgd_momentum_nesterov,
                dampening         = opt.sgd_momentum_nesterov and 0 or opt.sgd_momentum }
}

local optim_state = optim_states[opt.optim] or error('Unrecognised optim algorithm:'..opt.optim)
local optim_algo = optim[opt.optim]
print(opt.optim, optim_state)

local iterations = opt.max_epochs * #loader._train_batches
local iterations_per_epoch = loader.ntrain
local loss0 = nil
for i = 1, iterations do
    local epoch = i / #loader._train_batches

    local timer = torch.Timer()
    local _, loss = optim_algo(feval, params, optim_state)
    local time = timer:time().real

    local train_loss = math.exp(loss[1]) -- the loss is inside a list, pop it
    train_losses[i] = train_loss

    -- exponential learning rate decay
    if i % #loader._train_batches == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            if optim_state.learningRate then
               optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
               print('decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
            end
        end
    end

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
        -- evaluate loss on validation data
        local val_loss = math.exp(eval_split(loader:valid_batches()))
        val_losses[i] = val_loss

        local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', opt.checkpoint_dir, opt.savefile, epoch, val_loss)
        print('saving checkpoint to ' .. savefile)
        local checkpoint = {}
        checkpoint.protos = protos
        checkpoint.opt = opt
        checkpoint.train_losses = train_losses
        checkpoint.val_loss = val_loss
        checkpoint.val_losses = val_losses
        checkpoint.i = i
        checkpoint.epoch = epoch
        checkpoint.vocab = loader._word2class
        torch.save(savefile, checkpoint)
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), perplexity = %6.2f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_loss, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('loss is exploding, aborting.')
        break -- halt
    end
end


