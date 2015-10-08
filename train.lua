require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'autobw'

require 'util.misc'
require 'util.HLogSoftMax'
require 'util.Squeeze'

tbx = require 'pl.tablex'
HSMClass = require 'util.HSMClass'
LSM = require 'model.LSM'
text = require 'text'
text_sampler = require 'TextSampler'
model_utils = require 'util.model_utils'
gpu_utils = require 'util.gpu_utils'
initialiser = require 'util.initialiser'
LSTM = require 'model.LSTM'
GRU = require 'model.GRU'
RNN = require 'model.RNN'
IRNN = require 'model.IRNN'
SCRNN = require 'model.SCRNN'
CBOW = require 'model.CBOW'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a recurrent language model')
cmd:text()
cmd:text('Options')
-- data
cmd:option('-data_dir','data','data directory')
cmd:option('-ds_name', 'ptb', 'dataset name')
cmd:option('-vocab_size', 1e4, 'Number of words in the dictionary')
cmd:option('-seq_length', 50, 'Maximum number of tokens in a sentence')
cmd:option('-max_reps', 5, 'Maximum number of repeated tokens per line')
cmd:option('-s3_input', '', 'S3 dataset base location, / terminated')
cmd:option('-s3_output', '', 'S3 logs base location, / terminated')
-- model params
cmd:option('-rnn_size', 512, 'Size of recurrent internal state')
cmd:option('-context_size', 128, 'Size of SCRNN context state or of word context for CBOW')
cmd:option('-num_layers', 1, 'Number of recurrent layers')
cmd:option('-model', 'rnn', 'rnn | irnn | gru | lstm | scrnn | cbow')
cmd:option('-emb_size', 160, 'Size of word embeddings')
cmd:option('-hsm', 0, 'HSM classes, 0 is off, -1 is sqrt(vocab)')
cmd:option('-emb_sharing', 1, 'Share the encoder/decoder matrices, 1 is on')
cmd:option('-bias_init', 1, 'Initialise the softmax bias with unig probs, 1 is on')
-- optimization
cmd:option('-optim', 'adam', 'Optimisation algorithm')
cmd:option('-learning_rate', 1e-3, 'Initial learning rate')
cmd:option('-learning_rate_decay' ,0.9, 'Learning rate decay factor')
cmd:option('-ppl_tolerance', 2, 'Maximum difference between current PPL and best, triggers decaying')
cmd:option('-dropout', 0.5,'Dropout for regularization, 0 = no dropout')
cmd:option('-sgd_weight_decay', 0, 'SGD weight decay or L2 regularisation')
cmd:option('-sgd_momentum', 0, 'SGD momentum')
cmd:option('-sgd_momentum_nesterov', false, 'SGD, use nesterov momentum')
cmd:option('-rmsprop_alpha', 0.99,'RMSprop smoothing constant')
cmd:option('-rmsprop_epsilon', 1e-8, 'RMSprop epsilon')
cmd:option('-adam_beta1', 0.9, 'ADAM first moment coefficient')
cmd:option('-adam_beta2', 0.999, 'ADAM second moment coefficient')
cmd:option('-adam_lambda', 1e-8, 'ADAM first moment decay')
cmd:option('-adadelta_rho', 0.95, 'ADADELTA interpolation parameter')
cmd:option('-batch_size', 256, 'Number of sequences to train on in parallel')
cmd:option('-max_epochs', 1, 'Total number of full passes through the training data')
cmd:option('-max_valids', 0, 'Total number of evaluations, kind of max_epochs, 0 is off')
cmd:option('-grad_max_norm', 5, 'Renorm gradients at this value')
cmd:option('-init_from', '', 'Initialize network parameters from checkpoint at this path')
-- bookkeeping
cmd:option('-plot', false, 'Plot learning curves')
cmd:option('-seed', 42, 'Seed for random number generator, for repeatable experiments')
cmd:option('-print_every', 10, 'How many steps/minibatches between printing out the loss?')
cmd:option('-valid_period', 1e3, 'How many iterations between evaluating on validation data?')
cmd:option('-logs_dir', 'logs', 'Output root directory for experiment logs')
-- GPU/CPU
cmd:option('-gpuid', 0, 'Which gpu to use, -1 = use CPU')
cmd:option('-opencl', 0, 'Use OpenCL (instead of CUDA)')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- unique logs_dir
local exp_time = os.date('%Y.%m.%d_%H.%M')
opt.logs_dir = paths.concat(opt.logs_dir, opt.ds_name, exp_time)
paths.mkdir(opt.logs_dir)

-- loggers
local train_logger = optim.Logger(paths.concat(opt.logs_dir, 'train.log'))
local valid_logger = optim.Logger(paths.concat(opt.logs_dir, 'valid.log'))
train_logger:setNames{'train ppl'}
valid_logger:setNames{'train ppl', 'val ppl'}
train_logger:style{'-'}
valid_logger:style{'-', '-'}

-- print parmas
print('Parameters:')
print(opt)

-- seed
torch.manualSeed(opt.seed)

-- use FloatTensor when on CPU
torch.setdefaulttensortype('torch.FloatTensor')

-- check parameters
opt.emb_sharing = (opt.emb_sharing == 1) and true or false
opt.bias_init = (opt.bias_init == 1) and true or false
train_new_model = opt.init_from == ''
if opt.hsm ~= 0 then
   opt.bias_init = false
   if opt.emb_sharing then opt.hsm = opt.emb_size end
end

-- GPU?
gpu_utils.init()

-- fetch dataset from S3?
if opt.s3_input ~= '' then
   -- is it there?
   local found = sys.execute(string.format('aws s3 ls %s', opt.s3_input..opt.ds_name)) ~= ''
   if not found then error(string.format('Could not find datasest %s at location %s', opt.ds_name, opt.s3_input)) end
   -- sync it
   os.execute(string.format('aws s3 sync %s %s', opt.s3_input..opt.ds_name, paths.concat(opt.data_dir, opt.ds_name)))
   print(string.format('Dataset sychronised from S3, local directory: %s', paths.concat(opt.data_dir, opt.ds_name)))
end

-- create the Text loader class
print('Load text dataset...')
local text = Text{
   name = opt.ds_name,
   data_path = opt.data_dir,
   vocab_size = opt.vocab_size,
   batch_size = opt.batch_size,
   max_length = opt.seq_length,
   max_reps = opt.max_reps
}
-- samplr ignores other params when in rnn mode
local txt_sampler = TextSampler{mode=TextSampler.networkToMode(opt.model),
                                context_size=opt.context_size,
                                max_batch_size=text._batch_size,
                                max_length=text._max_length,
                                start_id=text:word2class()[text:start_sym()]}
local vocab_size = text:vocab_size()
local word2class = text:word2class()
if opt.hsm ~= 0 then text:setupHSM_alpha(opt.hsm) end
print(text)
--TODO print a 'text summary'

curr_batch_idx = 1
train_ppl_lst = {}
val_ppl_lst = {}

-- define the model: prototypes for one timestep, then clone them in time
if train_new_model then
   -- create new recurrent model prototype
   print(string.format('Creating a %s model with %s layers', opt.model:upper(), opt.num_layers))
   protos = {}
   if opt.model == 'lstm' then
      protos.rnn = LSTM.lstm(vocab_size, opt.rnn_size, opt.num_layers, opt.emb_size, opt.dropout, opt.hsm)
   elseif opt.model == 'gru' then
      protos.rnn = GRU.gru(vocab_size, opt.rnn_size, opt.num_layers, opt.emb_size, opt.dropout, opt.hsm)
   elseif opt.model == 'rnn' then
      protos.rnn = RNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.emb_size, opt.dropout, opt.hsm)
   elseif opt.model == 'irnn' then
      protos.rnn = IRNN.rnn(vocab_size, opt.rnn_size, opt.num_layers, opt.emb_size, opt.dropout, opt.hsm)
   elseif opt.model == 'scrnn' then
      protos.rnn = SCRNN.scrnn(vocab_size, opt.rnn_size, opt.context_size, opt.num_layers, opt.emb_size, opt.dropout, opt.hsm)
   elseif opt.model == 'cbow' then
       protos.rnn = CBOW.create(vocab_size, opt.context_size, vocab_size, opt.rnn_size, opt.emb_size, opt.dropout, opt.hsm)
   end
   -- HSM?
   if opt.hsm ~= 0 then
      protos.criterion = nn.HLogSoftMax(text:hsm_mapping(), opt.emb_size)
   else
      protos.criterion = nn.ClassNLLCriterion()
   end
else
   -- load previously trained model
   print(string.format('Loading model from checkpoint %s', opt.init_from))
   local checkpoint = torch.load(opt.init_from)
   -- make sure the checkpoint is compatible with new configuration
   assert(tbx.deepcompare(checkpoint.word2class, word2class), 'Error, word2class mismatch!')
   assert(opt.rnn_size == checkpoint.opt.rnn_size, 'Error, rnn_size was: '..checkpoint.opt.rnn_size)
   assert(opt.num_layers == checkpoint.opt.num_layers, 'Error! num_layers was: '..checkpoint.opt.num_layers)
   assert(opt.valid_period == checkpoint.opt.valid_period, 'Error! valid_period was: '.. checkpoint.opt.valid_period)
   -- fetch model and metadata
   protos = checkpoint.protos
   train_ppl_lst = checkpoint.train_ppl_lst
   val_ppl_lst = checkpoint.val_ppl_lst
   curr_batch_idx = checkpoint.batch_idx+1
   epoch = checkpoint.epoch
end

-- setup initial state sizes
init_state_sizes = {}
for i=1,opt.num_layers do
   if opt.model == 'scrnn' then
      table.insert(init_state_sizes, opt.context_size)
   end
   table.insert(init_state_sizes, opt.rnn_size)
   if opt.model == 'lstm' then
      table.insert(init_state_sizes, opt.rnn_size)
   end
end

-- softmax bias init?
if opt.bias_init then
   protos.rnn:apply(function(layer)
      if layer.name ~= nil and layer.name == 'lsm' then
         print('Initialising softmax bias to unigram distribution')
         layer.decoder.data.module.bias = text:unig_probs()
      end
   end)
end

-- ship the model to the GPU?
protos = gpu_utils.ship_table(protos)

-- share embeddings?
if opt.emb_sharing then
   model_utils.share_embeddings(protos)
end

-- put everything into one flattened parameters tensor
if opt.hsm == 0 then
   params, grad_params = model_utils.combine_all_parameters(protos.rnn)
else
   params, grad_params = model_utils.combine_all_parameters(protos.rnn, protos.criterion)
end
print('Total number of parameters in the model: ' .. params:nElement())

-- initialisation
if train_new_model then
   print('Initialising network weights')
   --params:uniform(-0.08, 0.08) -- small numbers uniform
   initialiser.initialise_network(protos.rnn, opt.model=='irnn')
   if opt.hsm ~= 0 then
      initialiser.initialise_network(protos.criterion, opt.model=='irnn')
   end
end

-- make a bunch of clones after flattening, as that reallocates memory
clones = {}
if opt.model == 'cbow' then
   print('Cloning fixed-context model but just the once')
   for name, proto in pairs(protos) do
      clones[name] = model_utils.clone_many_times(proto, 1, not proto.parameters)[1]
   end
else
   for name, proto in pairs(protos) do
      print(string.format('Cloning %s %s times', name, opt.seq_length+1))
      -- seq_length + 1 because Text prepends start of sequence token
      clones[name] = model_utils.clone_many_times(proto, opt.seq_length+1, not proto.parameters)
   end
end

-- evaluate the loss over on validation set
function run_validation()
   -- setup
   local tot_batches = #text:valid_batches()
   txt_sampler:reset_batch_pointer(text, text:valid_batches())
   local loss = 0
   xlua.progress(1,tot_batches)
   -- iterate over all valid batches
   for i=1,tot_batches do
      -- fetch a batch
      local inputs, targets = txt_sampler:next_batch(text,text:valid_batches())
      local curr_batch_size = inputs:size(1)
      local curr_seq_length = inputs:size(2)
      local init_state_local = {}
      for _, init_size in ipairs(init_state_sizes) do
         table.insert(init_state_local, torch.zeros(curr_batch_size, init_size))
      end
      -- ship to gpu?
      inputs = gpu_utils.ship(inputs)
      targets = gpu_utils.ship(targets)
      init_state_local = gpu_utils.ship_table(init_state_local)
      -- forward pass
      local rnn_state = {[0] = init_state_local}
      local curr_loss = 0
      for t=1, curr_seq_length do
         clones.rnn[t]:evaluate() -- for dropout proper functioning
         local lst = clones.rnn[t]:forward{inputs[{{}, t}], unpack(rnn_state[t-1])}
         rnn_state[t] = {}
         for i=1,#init_state_sizes do table.insert(rnn_state[t], lst[i]) end
         curr_loss = curr_loss + clones.criterion[t]:forward(lst[#lst], targets[{{}, t}])
      end
      -- update valid loss
      loss = loss + curr_loss / curr_seq_length
      xlua.progress(i, tot_batches)
      collectgarbage()
   end
   -- wrap up
   xlua.progress(tot_batches, tot_batches)
   loss = loss / tot_batches
   return loss
end

-- for fixed-context, feed-fwd NNs
function run_validation_fixed()
   -- setup
   local tot_batches = #text:valid_batches()
   txt_sampler:reset_batch_pointer(text, text:valid_batches())
   local loss = 0
   xlua.progress(1,tot_batches)
   -- iterate over all valid batches
   for i=1,tot_batches do
      -- fetch a batch
      local inputs, targets = txt_sampler:next_batch(text,text:valid_batches())
      local curr_batch_size = inputs:size(1)
      local curr_seq_length = inputs:size(2)
      -- ship to gpu?
      inputs = gpu_utils.ship(inputs)
      targets = gpu_utils.ship(targets)
      -- forward pass
      clones.rnn:evaluate() -- for dropout proper functioning
      local preds = clones.rnn:forward(inputs)
      -- update valid loss
      loss = loss + clones.criterion:forward(preds, targets)
      xlua.progress(i, tot_batches)
      collectgarbage()
   end
   -- wrap up
   xlua.progress(tot_batches, tot_batches)
   loss = loss / tot_batches
   return loss
end

-- do fwd/bwd and return loss, grad_params
function feval(x)
   if x ~= params then
      params:copy(x)
   end
   grad_params:zero()
   ------------------ get minibatch -------------------
   local inputs, targets = txt_sampler:next_batch(text,text:train_batches())
   print('inputs')
   print(inputs[{{}, 1}]:size())
   print('targets')
   print(targets[{{}, 1}]:size())
   local curr_batch_size = inputs:size(1)
   local curr_seq_length = inputs:size(2)
   local init_state_local = {}
   for _, init_size in ipairs(init_state_sizes) do
      table.insert(init_state_local, torch.zeros(curr_batch_size, init_size))
   end
   -- ship to gpu?
   inputs = gpu_utils.ship(inputs)
   targets = gpu_utils.ship(targets)
   init_state_local = gpu_utils.ship_table(init_state_local)
   ------------------- forward pass -------------------
   local tape = autobw.Tape()
   tape:begin()
   local rnn_state = {[0] = init_state_local}
   local loss = 0
   local ts_timer = torch.Timer()
   for t=1,curr_seq_length do
      clones.rnn[t]:training() -- make sure we are in correct mode (this is cheap, sets flag)
      local lst = clones.rnn[t]:forward{inputs[{{}, t}], unpack(rnn_state[t-1])}
      rnn_state[t] = {}
      for i=1,#init_state_sizes do table.insert(rnn_state[t], lst[i]) end -- extract the state, without output
      loss = loss + clones.criterion[t]:forward(lst[#lst], targets[{{}, t}])
   end
   loss = loss / curr_seq_length
   tape:stop()
   ------------------ backward pass -------------------
   tape:backward()
   ------------------------ misc ----------------------
   -- renorm gradients
   local gp_norm = grad_params:norm()
   if gp_norm > opt.grad_max_norm then
      grad_params:mul(opt.grad_max_norm / gp_norm)
      -- print(string.format('grads renorm: %.2f -> %.2f', gp_norm, grad_params:norm()))
   end
   -- get average token/seconds
   local curr_ts = inputs:nElement() / ts_timer:time().real
   avg_ts = avg_ts and 0.99 * avg_ts + 0.01 * curr_ts or curr_ts
   -- TODO pass on final context state to next batch in SCRNN
   return loss, grad_params
end

-- feval for fixed-context, feed-fwd NNs
function feval_fixed(x)
   if x ~= params then
      params:copy(x)
   end
   grad_params:zero()
   ------------------ get minibatch -------------------
   local inputs, targets = txt_sampler:next_batch(text,text:train_batches())
   local curr_batch_size = inputs:size(1)
   local curr_seq_length = inputs:size(2)
   -- ship to gpu?
   inputs = gpu_utils.ship(inputs)
   targets = gpu_utils.ship(targets)
   ------------------- forward pass -------------------
   local tape = autobw.Tape()
   tape:begin()
   local loss = 0
   local ts_timer = torch.Timer()
   clones.rnn:training() -- make sure we are in correct mode (this is cheap, sets flag)
   local preds = clones.rnn:forward(inputs)
   loss = loss + clones.criterion:forward(preds, targets)
   tape:stop()
   ------------------ backward pass -------------------
   tape:backward()
   ------------------------ misc ----------------------
   -- renorm gradients
   local gp_norm = grad_params:norm()
   if gp_norm > opt.grad_max_norm then
      grad_params:mul(opt.grad_max_norm / gp_norm)
      -- print(string.format('grads renorm: %.2f -> %.2f', gp_norm, grad_params:norm()))
   end
   -- get average token/seconds
   local curr_ts = inputs:nElement() / ts_timer:time().real
   avg_ts = avg_ts and 0.99 * avg_ts + 0.01 * curr_ts or curr_ts
   return loss, grad_params
end

-- start optimization here
local optim_states = {
   rmsprop  = { learningRate = opt.learning_rate,
                alpha        = opt.rmsprop_alpha,
                epsilon      = opt.rmsprop_epsilon },
   adagrad  = { learningRate      = opt.learning_rate,
                learningRateDecay = opt.learning_rate_decay },
   adam     = { learningRate = opt.learning_rate,
                beta1  = opt.adam_beta1,
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

print('Starting training!')
local iterations = opt.max_epochs * #text:train_batches()
local loss0 = nil
local best_ppl = nil
local total_valids = 0
txt_sampler:reset_batch_pointer(text, text:train_batches(), curr_batch_idx, batch_idx) -- (re)set train batch pointer
local valid_fn = opt.model == 'cbow' and run_validation_fixed or run_validation
local feval_fn = opt.model == 'cbow' and feval_fixed or feval
-- TRAIN loop
for batch_idx=curr_batch_idx, iterations do
   -- setup
   local timer = torch.Timer()
   local epoch = batch_idx / #text:train_batches()
   local _, loss = optim_algo(feval_fn, params, optim_state)
   local time = timer:time().real
   -- save train ppl
   train_ppl = math.exp(loss[1]) -- the loss is inside a list, pop it
   table.insert(train_ppl_lst, train_ppl)
   -- every now and then or on last iteration
   if batch_idx % opt.valid_period == 0 or batch_idx == iterations then
      -- validation
      local val_ppl = math.exp(valid_fn(text:valid_batches()))
      val_ppl_lst[batch_idx] = val_ppl
      print('Evaluation PPL: '..val_ppl)
      -- plot?
      valid_logger:add{train_ppl, val_ppl}
      if opt.plot then valid_logger:plot() end
      -- best model? -> save!
      if best_ppl == nil or val_ppl < best_ppl then
         best_ppl = val_ppl
         local savefile = paths.concat(opt.logs_dir, string.format('model_%s.t7', exp_time))
         print('Saving checkpoint to ' .. savefile)
         local checkpoint = {}
         checkpoint.protos = gpu_utils.to_ram_table(protos)
         checkpoint.opt = opt
         checkpoint.train_ppl_lst = train_ppl_lst
         checkpoint.val_ppl = val_ppl
         checkpoint.val_ppl_lst = val_ppl_lst
         checkpoint.batch_idx = batch_idx
         checkpoint.epoch = epoch
         checkpoint.word2class = text:word2class()
         torch.save(savefile, checkpoint)
         -- upload to S3?
         if opt.s3_output ~= '' then
            print(string.format('Syncronising local log directory %s to S3 %s', opt.logs_dir, opt.s3_output))
            os.execute(string.format('aws s3 sync %s %s', opt.logs_dir, opt.s3_output..opt.ds_name..'/'..exp_time))
         end
         -- all done?
         if opt.max_valids > 0 and total_valids == opt.max_valids then
            print('Reached the end of the journey, hope you enjoyed!')
            os.exit()
         end
      end
      -- reduce the learning rate?
      if best_ppl and (val_ppl - best_ppl) > opt.ppl_tolerance then
         if optim_state.learningRate then
            local new_lr = optim_state.learningRate * opt.learning_rate_decay
            print(string.format('PPL not dropping! %.2f >> %.2f, reducing learning rate: %s -> %s', val_ppl, best_ppl, optim_state.learningRate, new_lr))
            optim_state.learningRate = new_lr
         end
      end
   end
   -- print train stats?
   if batch_idx % opt.print_every == 0 then
      print(string.format("%d/%d (epoch %.3f), perplexity: %7.2f, grad/param: %5.4e, tokens/sec: %.f", batch_idx, iterations, epoch, train_ppl, grad_params:norm()/params:norm(), avg_ts))
      -- plot?
      train_logger:add{train_ppl}
      if opt.plot then train_logger:plot() end
   end
   -- handle early stopping if things are going really bad
   if loss[1] ~= loss[1] then
      print('Loss is NaN.  This usually indicates a bug.  Please check the issues page for existing issues, or create a new issue, if none exist.  Ideally, please state: your operating system, 32-bit/64-bit, your blas version, cpu/cuda/cl?')
      break -- halt
   end
   -- loss exploding?
   if loss0 == nil then loss0 = loss[1] end
   if loss[1] > loss0 * 5 then
      print('Loss is exploding, aborting.')
      break -- halt
   end
   -- gc
   if batch_idx % 100 == 0 then collectgarbage() end
end
