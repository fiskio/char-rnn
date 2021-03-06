
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

require 'util.OneHot'
require 'util.misc'
local CharSplitLMMinibatchLoader = require 'util.CharSplitLMMinibatchLoader'
local model_utils = require 'util.model_utils'
local LSTM = require 'model.LSTM'
local GRU = require 'model.GRU'
local RNN = require 'model.RNN'
local IRNN = require 'model.IRNN'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a character-level language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data/tinyshakespeare','data directory. Should contain the file input.txt with input data')
-- model params
cmd:option('-rnn_size', 100, 'size of LSTM internal state')
cmd:option('-num_layers', 2, 'number of layers in the LSTM')
cmd:option('-model', 'lstm', 'for now only lstm is supported. keep fixed')
-- optimization
cmd:option('-optim', 'rmsprop', 'Optimisation algorithm')
cmd:option('-learning_rate',2e-3,'learning rate')
cmd:option('-learning_rate_decay', 0, 'learning rate decay')
cmd:option('-sgd_weight_decay', 0, 'weight decay')
cmd:option('-sgd_momentum', 0, 'momentum')
cmd:option('-sgd_momentum_nesterov', false, 'use nesterov momentum ')
cmd:option('-rmsprop_alpha', 0.99,'smoothing constant')
cmd:option('-rmsprop_epsilon', 1e-8, 'value with which to inistialise m')
cmd:option('-adam_beta1', 0.9, 'first moment coefficient')
cmd:option('-adam_beta2', 0.999, 'second moment coefficient')
cmd:option('-adam_lambda', 1-1e-8, 'first moment decay')
cmd:option('-adadelta_rho', 0.9, 'interpolation parameter')
cmd:option('-dropout',0,'dropout to use just before classifier. 0 = no dropout')
cmd:option('-seq_length',50,'number of timesteps to unroll for')
cmd:option('-batch_size',100,'number of sequences to train on in parallel')
cmd:option('-max_epochs',30,'number of full passes through the training data')
cmd:option('-grad_clip',5,'clip gradients at')
cmd:option('-train_frac',0.95,'fraction of data that goes into train set')
cmd:option('-val_frac',0.05,'fraction of data that goes into validation set')
            -- note: test_frac will be computed as (1 - train_frac - val_frac)
-- bookkeeping
cmd:option('-seed',123,'torch manual random number generator seed')
cmd:option('-print_every',10,'how many steps/minibatches between printing out the loss')
cmd:option('-eval_val_every',1000,'every how many iterations should we evaluate on validation data?')
cmd:option('-checkpoint_dir', 'cv', 'output directory where checkpoints get written')
cmd:option('-savefile', '','filename to autosave the checkpont to. Will be inside checkpoint_dir/')
cmd:option('-plot', false, 'plot training and validation bits per charater')
-- GPU/CPU
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
-- train / val / test split for data, in fractions
local test_frac = math.max(0, 1 - opt.train_frac - opt.val_frac)
local split_sizes = {opt.train_frac, opt.val_frac, test_frac}

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
-- create the data loader class
local loader = CharSplitLMMinibatchLoader.create(opt.data_dir, opt.batch_size, opt.seq_length, split_sizes)
local vocab_size = loader.vocab_size  -- the number of distinct characters
print('vocab size: ' .. vocab_size)
-- make sure output directory exists
local checkpoint_dir = paths.concat(opt.checkpoint_dir, paths.basename(opt.data_dir))
if not path.exists(checkpoint_dir) then lfs.mkdir(checkpoint_dir) end

-- define the model: prototypes for one timestep, then clone them in time
protos = {}
protos.embed = OneHot(vocab_size)

-- which units?
if opt.model == 'lstm' then
   protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.dropout)
elseif opt.model == 'gru' then
   protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers)
elseif opt.model == 'rnn' then
   protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers)
elseif opt.model == 'irnn' then
   protos.rnn = IRNN.irnn(vocab_size, opt.rnn_size, opt.num_layers)
else
   error('unsupported recurrent units: '..opt.model)
end
print('creating a '..opt.model:upper()..' with '..opt.num_layers..' layers')

-- the initial state of the cell/hidden states
init_state = {}
for L=1,opt.num_layers do
    local h_init = torch.zeros(opt.batch_size, opt.rnn_size)
    if opt.gpuid >=0 then h_init = h_init:cuda() end
    table.insert(init_state, h_init:clone())
    -- lstm requires 2 initial states
    if opt.model == 'lstm' then
       table.insert(init_state, h_init:clone())
    end
end
state_predict_index = #init_state -- index of blob to make prediction from
-- classifier on top
protos.softmax = nn.Sequential():add(nn.Linear(opt.rnn_size, vocab_size)):add(nn.LogSoftMax())
-- training criterion (negative log likelihood)
protos.criterion = nn.ClassNLLCriterion()

-- ship the model to the GPU if desired
if opt.gpuid >= 0 then
    for k,v in pairs(protos) do v:cuda() end
end

-- put the above things into one flattened parameters tensor
params, grad_params = model_utils.combine_all_parameters(protos.embed, protos.rnn, protos.softmax)
params:uniform(-0.08, 0.08)
print('number of parameters in the model: ' .. params:nElement())
-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
for name,proto in pairs(protos) do
    print('cloning ' .. name)
    clones[name] = model_utils.clone_many_times(proto, opt.seq_length, not proto.parameters)
end

-- evaluate the loss over an entire split
function eval_split(split_index, max_batches)
    print('evaluating loss over split index ' .. split_index)
    local n = loader.split_sizes[split_index]
    if max_batches ~= nil then n = math.min(max_batches, n) end

    loader:reset_batch_pointer(split_index) -- move batch iteration pointer for this split to front
    local loss = 0
    local rnn_state = {[0] = init_state}

    for i = 1,n do -- iterate over batches in the split
        -- fetch a batch
        local x, y = loader:next_batch(split_index)
        if opt.gpuid >= 0 then -- ship the input arrays to GPU
            -- have to convert to float because integers can't be cuda()'d
            x = x:float():cuda()
            y = y:float():cuda()
        end
        -- forward pass
        for t=1,opt.seq_length do
            local embedding = clones.embed[t]:forward(x[{{}, t}])
            clones.rnn[t]:evaluate() -- for dropout proper functioning
            rnn_state[t] = clones.rnn[t]:forward{embedding, unpack(rnn_state[t-1])}
            if type(rnn_state[t]) ~= 'table' then rnn_state[t] = {rnn_state[t]} end
            local prediction = clones.softmax[t]:forward(rnn_state[t][state_predict_index])
            loss = loss + clones.criterion[t]:forward(prediction, y[{{}, t}])
        end
        -- carry over lstm state
        rnn_state[0] = rnn_state[#rnn_state]
        xlua.progress(i, n)
    end

    loss = loss / opt.seq_length / n
    return loss
end

-- do fwd/bwd and return loss, grad_params
local init_state_global = clone_list(init_state)
function feval(x)
    if x ~= params then
        params:copy(x)
    end
    grad_params:zero()

    ------------------ get minibatch -------------------
    local x, y = loader:next_batch(1)
    if opt.gpuid >= 0 then -- ship the input arrays to GPU
        -- have to convert to float because integers can't be cuda()'d
        x = x:float():cuda()
        y = y:float():cuda()
    end
    ------------------- forward pass -------------------
    local embeddings = {}            -- input embeddings
    local rnn_state = {[0] = init_state_global}
    local predictions = {}           -- softmax outputs
    local loss = 0
    for t=1,opt.seq_length do
        embeddings[t] = clones.embed[t]:forward(x[{{}, t}])
        clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
        rnn_state[t] = clones.rnn[t]:forward{embeddings[t], unpack(rnn_state[t-1])}
        -- the following line is needed because nngraph tries to be clever
        if type(rnn_state[t]) ~= 'table' then rnn_state[t] = {rnn_state[t]} end
        predictions[t] = clones.softmax[t]:forward(rnn_state[t][state_predict_index])
        loss = loss + clones.criterion[t]:forward(predictions[t], y[{{}, t}])
    end
    loss = loss / opt.seq_length
    ------------------ backward pass -------------------
    local dembeddings = {}
    -- initialize gradient at time t to be zeros (there's no influence from future)
    local drnn_state = {[opt.seq_length] = clone_list(init_state, true)} -- true also zeros the clones
    for t=opt.seq_length,1,-1 do
        -- backprop through loss, and softmax/linear
        local doutput_t = clones.criterion[t]:backward(predictions[t], y[{{}, t}])
        drnn_state[t][state_predict_index] = clones.softmax[t]:backward(rnn_state[t][state_predict_index], doutput_t)
        -- backprop through LSTM timestep
        local drnn_statet_passin = drnn_state[t]
        -- we have to be careful with nngraph again
        if #(rnn_state[t]) == 1 then drnn_statet_passin = drnn_state[t][1] end
        local dlst = clones.rnn[t]:backward({embeddings[t], unpack(rnn_state[t-1])}, drnn_statet_passin)
        drnn_state[t-1] = {}
        for k,v in pairs(dlst) do
            if k == 1 then
                dembeddings[t] = v
            else
                -- note we do k-1 because first item is dembeddings, and then follow the
                -- derivatives of the state, starting at index 2. I know...
                drnn_state[t-1][k-1] = v
            end
        end
        -- backprop through embeddings
        clones.embed[t]:backward(x[{{}, t}], dembeddings[t])
    end
    ------------------------ misc ----------------------
    -- transfer final state to initial state (BPTT)
    init_state_global = rnn_state[#rnn_state] -- NOTE: I don't think this needs to be a clone, right?
    -- clip gradient element-wise
    grad_params:clamp(-opt.grad_clip, opt.grad_clip)
    return loss, grad_params
end

-- start optimization here
local optim_states = {

   ['rmsprop'] = { learningRate = opt.learning_rate, alpha = opt.rmsprop_alpha, epsilon = opt.rmsprop_epsilon },
   ['adagrad'] = { learningRate = opt.learning_rate, learningRateDecay = opt.learning_rate_decay },
   ['adam'] = { beta1 = opt.adam_beta1, beta2 = opt.adam_beta2, lambda = opt.adam_lambda },
   ['adadelta'] = { rho = opt.adadelta_rho },
   ['sgd'] = { learningRate = opt.learning_rate,
               learningRateDecay = opt.learning_rate_decay,
               weightDecay = opt.sgd_weight_decay,
               momentum = opt.sgd_momentum,
               nesterov = opt.sgd_momentum_nesterov,
               dampening = opt.sgd_momentum_nesterov and 0 or opt.sgd_momentum }
}
train_bpcs = {} -- bits per character
val_bpcs = {}
-- setup optimisation algo
local optim_state = optim_states[opt.optim] or error('Unrecognised optim algorithm:'..opt.optim)
local optim_algo = optim[opt.optim]
print(opt.optim, optim_state)
local iterations = opt.max_epochs * loader.ntrain
local iterations_per_epoch = loader.ntrain
local loss0 = nil
local bmt
-- setup logger
local log_dir = 'log/'..paths.basename(opt.data_dir)..'/'
local log_pfx = opt.model..'-'..opt.rnn_size..'x'..opt.num_layers..'.'..opt.optim
local train_logger = optim.Logger(log_dir..log_pfx..'.train.log')
local valid_logger = optim.Logger(log_dir..log_pfx..'.valid.log')
train_logger:setNames{'train_bpc'}
valid_logger:setNames{'train_bpc', 'val_bpc'}
train_logger:style{'-'}
valid_logger:style{'-', '-'}

local gtimer = torch.Timer()
local etimer = torch.Timer()
local btimer = torch.Timer()
local vtimer = torch.Timer()

for e = 1, opt.max_epochs do
   etimer:reset()
   for b = 1, loader.ntrain do
      local i = ((e-1) * loader.ntrain) + b
      local epoch = i / loader.ntrain

      btimer:reset()
      local _, loss = optim_algo(feval, params, optim_state)
      local time = btimer:time().real
      bmt = bmt and 0.99 * bmt + 0.01 * time or time

      local train_bpc = math.log(math.exp(loss[1]),2) -- exp to get probability, then log 2 it to get bits per character
      train_bpcs[i] = train_bpc

      -- every now and then or on last iteration
      if i % opt.eval_val_every == 0 or i == iterations then
         -- evaluate bpc on validation data
         vtimer:reset()
         local val_loss = eval_split(2) -- 2 = validation
         val_bpc = math.log(math.exp(val_loss),2)
         print(string.format('Validation took: %s - vbpc: %.2f', os.date("!%X", vtimer:time().real), val_bpc))
         val_bpcs[i] = val_bpc
         valid_logger:add{ train_bpc, val_bpc }
         if opt.plot then valid_logger:plot() end
         if not best_val_bpc or val_bpc < best_val_bpc then
            local savekey = opt.savefile == '' and log_pfx or opt.savefile
            local savefile = string.format('%s/lm_%s_epoch%.2f_%.4f.t7', checkpoint_dir, savekey, epoch, val_bpc)
            print('saving checkpoint to ' .. savefile)
            local checkpoint = {}
            checkpoint.protos = protos
            checkpoint.opt = opt
            checkpoint.train_bpcs = train_bpcs
            checkpoint.val_bpc = val_bpc
            checkpoint.val_bpc = val_bpc
            checkpoint.i = i
            checkpoint.epoch = epoch
            checkpoint.vocab = loader.vocab_mapping
            torch.save(savefile, checkpoint)
            if current_best then os.remove(current_best) end
            current_best = savefile
            best_val_bpc = val_bpc
         else
            print(string.format('val_bpc has increased, not saving %.2f > %.2f', val_bpc, best_val_bpc))
         end
      end

      if i % opt.print_every == 0 then
         local progress_perc = 100 * i / iterations
         local eta = os.date("!%X", bmt * (iterations - i))
         local norms_ratio = grad_params:norm() / params:norm()
         train_logger:add{ train_bpc }
         if opt.plot then train_logger:plot() end
         print(string.format("%.1f%% %d/%d (epoch %.3f), train_bpc = %6.8f, grad/param norm = %6.4e, time/batch = %.2fs, eta = %s", progress_perc, i, iterations, epoch, train_bpc, norms_ratio, time, eta))
      end

      if i % 10 == 0 then collectgarbage() end

      -- handle early stopping if things are going really bad
      if loss0 == nil then loss0 = loss[1] end
      if loss[1] > loss0 * 3 then
         print('loss is exploding, aborting.')
         break -- halt
      end
   end
   print(string.format('Epoch %d took: %s', e, os.date("!%X", etimer:time().real)))
end
print(string.format('Training took: %s', os.date("!%X", gtimer:time().real)))
