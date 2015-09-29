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
cmd:option('-vocab_size', 1e4, 'Number of words in the dictionary')
cmd:option('-seq_length', 50, 'Maximum number of tokens in a sentence')
cmd:option('-max_reps', 5, 'Maximum number of repeated tokens per line')
-- model params
cmd:option('-hidden_size', 512, 'Size of recurrent internal state')
cmd:option('-context_size', 128, 'Size of SCRNN context state')
cmd:option('-num_layers', 1, 'Number of recurrent layers')
cmd:option('-model', 'rnn', 'rnn | gru | lstm | scrnn')
cmd:option('-emb_size', 128, 'Size of word embeddings')
cmd:option('-hsm', 0, 'HSM classes, 0 is off, -1 is sqrt(vocab)')
cmd:option('-emb_sharing', true, 'Share the encoder/decoder matrices')
-- optimization
cmd:option('-optim', 'rmsprop', 'Optimisation algorithm')
cmd:option('-learning_rate', 1e-3, 'Initial learning rate')
cmd:option('-learning_rate_decay' ,0.95, 'Learning rate decay factor')
cmd:option('-ppl_tolerance', 5, 'Maximum difference between current PPL and best, triggers decaying')
--cmd:option('-learning_rate_decay_after',25,'in number of epochs, when to start decaying the learning rate')
cmd:option('-dropout', 0.5,'Dropout for regularization, 0 = no dropout')
cmd:option('-sgd_weight_decay', 0, 'SGD weight decay or L2 regularisation')
cmd:option('-sgd_momentum', 0, 'SGD momentum')
cmd:option('-sgd_momentum_nesterov', false, 'SGD, use nesterov momentum')
cmd:option('-rmsprop_alpha', 0.99,'RMSprop smoothing constant')
cmd:option('-rmsprop_epsilon', 1e-8, 'RMSprop epsilon')
cmd:option('-adam_beta1', 0.9, 'ADAM first moment coefficient')
cmd:option('-adam_beta2', 0.999, 'ADAM second moment coefficient')
cmd:option('-adam_lambda', 1-1e-8, 'ADAM first moment decay')
cmd:option('-adadelta_rho', 0.95, 'ADADELTA interpolation parameter')
cmd:option('-batch_size', 64, 'Number of sequences to train on in parallel')
cmd:option('-max_epochs', 50, 'Total number of full passes through the training data')
cmd:option('-grad_clip', 5, 'Clip gradients at this value')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-seed', 42, 'Seed for random number generator, for repeatable experiments')
cmd:option('-print_every', 10, 'How many steps/minibatches between printing out the loss?')
cmd:option('-eval_val_every', 1e3, 'How many iterations between evaluating on validation data?')
cmd:option('-logs_dir', 'logs', 'Output root directory for experiment logs')
-- GPU/CPU
cmd:option('-gpuid', 0, 'Which gpu to use, -1 = use CPU')
cmd:option('-opencl', 0, 'Use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- unique logs_dir
exp_time = os.date('%Y.%m.%d_%H.%M')
opt.logs_dir = paths.concat(opt.logs_dir, exp_time)
paths.mkdir(opt.logs_dir)

-- print parmas
print('Parameters:')
print(opt)

-- seed
torch.manualSeed(opt.seed)

-- check parameters
if opt.hsm ~= 0 and opt.emb_sharing then
   error('Sharing encoder/decoder matrices and HSM are incompatible!')
end

-- initialize cunn/cutorch for training on the GPU and fall back to CPU gracefully
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then print('Package cunn not found!') end
    if not ok2 then print('Package cutorch not found!') end
    if ok and ok2 then
        print('Using CUDA on GPU ' .. opt.gpuid .. '...')
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
    if not ok then print('Package clnn not found!') end
    if not ok2 then print('Package cltorch not found!') end
    if ok and ok2 then
        print('Using OpenCL on GPU ' .. opt.gpuid .. '...')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
        print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
        print('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- create the Text loader class
local loader = Text{
                  name = paths.basename(opt.data_dir),
                  data_path = paths.dirname(opt.data_dir),
                  vocab_size = opt.vocab_size,
                  batch_size = opt.batch_size,
                  max_length = opt.seq_length,
                  max_reps = opt.max_reps
               }
local vocab_size = loader._vocab_size  -- the number of distinct characters
local vocab = loader._word2class
--TODO print a 'text summary'
--
-- make sure log directory exists
if not path.exists(opt.logs_dir) then lfs.mkdir(opt.logs_dir) end

-- define the model: prototypes for one timestep, then clone them in time
local do_random_init = true
if string.len(opt.init_from) > 0 then
    print('Loading an LSTM from checkpoint ' .. opt.init_from)
    local checkpoint = torch.load(opt.init_from)
    protos = checkpoint.protos
    -- make sure the vocabs are the same
    local vocab_compatible = true
    for c,i in pairs(checkpoint.vocab) do
        if not vocab[c] == i then
            vocab_compatible = false
        end
    end
    assert(vocab_compatible, 'Error, the character vocabulary for this dataset and the one in the saved checkpoint are not the same. This is trouble.')
    -- overwrite model settings based on checkpoint to ensure compatibility
    print('Overwriting hidden_size=' .. checkpoint.opt.hidden_size .. ', num_layers=' .. checkpoint.opt.num_layers .. ' based on the checkpoint.')
    opt.hidden_size = checkpoint.opt.hidden_size
    opt.num_layers = checkpoint.opt.num_layers
    do_random_init = false
else
    print('Creating an ' .. opt.model .. ' with ' .. opt.num_layers .. ' layers')
    protos = {}
    if opt.model == 'lstm' then
        protos.rnn = LSTM.lstm(vocab_size, opt.hidden_size, opt.num_layers, opt.emb_size, opt.dropout)
    elseif opt.model == 'gru' then
        protos.rnn = GRU.gru(vocab_size, opt.hidden_size, opt.num_layers, opt.emb_size, opt.dropout)
    elseif opt.model == 'rnn' then
        protos.rnn = RNN.rnn(vocab_size, opt.hidden_size, opt.num_layers, opt.emb_size, opt.dropout, opt.hsm, opt.emb_sharing)
    elseif opt.model == 'scrnn' then
        protos.rnn = SCRNN.scrnn(vocab_size, opt.emb_size, opt.hidden_size, opt.context_size, opt.num_layers, opt.dropout)
    end
    -- HSM?
    if opt.hsm ~= 0 then
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

-- put everything into one flattened parameters tensor
if opt.hsm == 0 then
   params, grad_params = model_utils.combine_all_parameters(protos.rnn)
else
   params, grad_params = model_utils.combine_all_parameters(protos.rnn, protos.criterion)
end
print('Total number of parameters in the model: ' .. params:nElement())

-- initialization
if do_random_init then
   -- TODO improve initialisation
   params:uniform(-0.08, 0.08) -- small numbers uniform
end

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name, proto in pairs(protos) do
    print(string.format('Cloning %s %s times', name, opt.seq_length+1))
    -- seq_length + 1 because Text prepends start of sequence token
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length+1, not proto.parameters)
end

-- evaluate the loss over on validation set
function eval_split()
    local n = #loader._valid_batches
    loader:reset_batch_pointer(loader:valid_batches()) -- reset validation batch pointer
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
            -- have to convert to float because integers can't be cuda()
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
    -- TODO pass on context state in SCRNN
    -- init_state_global = rnn_state[#rnn_state]
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    -- TODO renorm?
    return loss, grad_params
end

-- start optimization here
train_ppls = {}
val_ppls = {}

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
print('Optimisation:')
print(opt.optim, optim_state)

local iterations = opt.max_epochs * #loader._train_batches
local iterations_per_epoch = loader.ntrain
local loss0 = nil
local best_ppl = nil
for i = 1, iterations do
    local epoch = i / #loader._train_batches

    local timer = torch.Timer()
    local _, loss = optim_algo(feval, params, optim_state)
    local time = timer:time().real

    local train_ppl = math.exp(loss[1]) -- the loss is inside a list, pop it
    train_ppls[i] = train_ppl

    --[[
    -- exponential learning rate decay
    if i % #loader._train_batches == 0 and opt.learning_rate_decay < 1 then
        if epoch >= opt.learning_rate_decay_after then
            local decay_factor = opt.learning_rate_decay
            if optim_state.learningRate then
               optim_state.learningRate = optim_state.learningRate * decay_factor -- decay it
               print('Decayed learning rate by a factor ' .. decay_factor .. ' to ' .. optim_state.learningRate)
            end
        end
    end
    --]]

    -- every now and then or on last iteration
    if i % opt.eval_val_every == 0 or i == iterations then
       -- evaluate loss on validation data
       local val_ppl = math.exp(eval_split(loader:valid_batches()))
       val_ppls[i] = val_ppl
       print('Evaluation perplexity: '..val_ppl)
       -- best model? -> save!
       if best_ppl == nil or val_ppl < best_ppl then
          best_ppl = val_ppl
          local savefile = paths.concat(opt.logs_dir, string.format('model_%s.t7', exp_time))
          print('Saving checkpoint to ' .. savefile)
          local checkpoint = {}
          checkpoint.protos = protos
          checkpoint.opt = opt
          checkpoint.train_ppls = train_ppls
          checkpoint.val_ppl = val_ppl
          checkpoint.val_losses = val_pples
          checkpoint.i = i
          checkpoint.epoch = epoch
          checkpoint.vocab = loader._word2class
          torch.save(savefile, checkpoint)
       end
       -- anneal the learning rate?
       if best_ppl and (val_ppl - best_ppl) > opt.ppl_tolerance then
          if optim_state.learningRate then
             local new_lr = optim_state.learningRate * opt.learning_rate_decay
             print(string.format('PPL not dropping! %.2f >> %.2f, reducing learning rate: %s -> %s', val_ppl, best_ppl, optim_state.learningRate, new_lr))
             optim_state.learningRate = new_lr
          end
       end
    end

    if i % opt.print_every == 0 then
        print(string.format("%d/%d (epoch %.3f), perplexity = %6.2f, grad/param norm = %6.4e, time/batch = %.2fs", i, iterations, epoch, train_ppl, grad_params:norm() / params:norm(), time))
    end

    if i % 10 == 0 then collectgarbage() end

    -- handle early stopping if things are going really bad
    if loss[1] ~= loss[1] then
        print('Loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')

        break -- halt
    end
    if loss0 == nil then loss0 = loss[1] end
    if loss[1] > loss0 * 3 then
        print('Loss is exploding, aborting.')
        break -- halt
    end
end


