
--[[

This file samples characters from a trained model

Code is based on implementation in 
https://github.com/oxford-cs-ml-2015/practical6

]]--

require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-argmax',false,'use max at each timestep, equivalent to -topK 1')
cmd:option('-primetext'," ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-sentence',false,'generate characters until a full stop')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-topK',0,'number of prediction to sample from')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)


local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

protos = checkpoint.protos
local rnn_idx = #protos.softmax.modules - 1
opt.rnn_size = protos.softmax.modules[rnn_idx].weight:size(2)

-- initialize the rnn state
local current_state, state_predict_index
local model = checkpoint.opt.model

print('creating a '..model:upper()..'...')
local num_layers = checkpoint.opt.num_layers or 1 -- or 1 is for backward compatibility
current_state = {}
for L=1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(current_state, h_init:clone())
    if model == 'lstm' then
       table.insert(current_state, h_init:clone())
    end
end
state_predict_index = #current_state -- last one is the top h
local seed_text = opt.primetext
local prev_char

protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- do a few seeded timesteps
print('')
io.write(sys.COLORS.green .. seed_text..'\027[00m')
for c in seed_text:gmatch'.' do
    prev_char = torch.Tensor{vocab[c]}
    if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
    local embedding = protos.embed:forward(prev_char)
    current_state = protos.rnn:forward{embedding, unpack(current_state)}
    if type(current_state) ~= 'table' then current_state = {current_state} end
end

-- start sampling/argmaxing
for i=1, opt.length do

    -- softmax from previous timestep
    local next_h = current_state[state_predict_index]
    next_h = next_h / opt.temperature
    local log_probs = protos.softmax:forward(next_h)

    if opt.argmax then
        -- use argmax
        local _, prev_char_ = log_probs:max(2)
        prev_char = prev_char_:resize(1)
    else
        -- use sampling
        local probs = torch.exp(log_probs):squeeze()
        local probs, ids = torch.sort(probs, 1, true)
        if opt.topK > probs:size(1) then error('-topK max value: '..probs:size(1)) end
        if opt.topK > 0 then
           probs = probs:sub(1, opt.topK)
           ids = ids:sub(1, opt.topK)
        end
        prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
        prev_char = ids[prev_char[1]]
        prev_char = torch.FloatTensor{prev_char}
    end

    -- forward the rnn for next character
    local embedding = protos.embed:forward(prev_char)
    current_state = protos.rnn:forward{embedding, unpack(current_state)}
    if type(current_state) ~= 'table' then current_state = {current_state} end
    local current_char = ivocab[prev_char[1]]
    io.write(current_char)
    -- full sentence?
    if opt.sentence and current_char:find('%.') then break; end
end
io.write('\n') io.flush()

