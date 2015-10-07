
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

model_utils = require 'util.model_utils'
gpu_utils = require 'util.gpu_utils'
initialiser = require 'util.initialiser'
emoji = require 'util.emoji'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Predict terms from a language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-npredictions',10,'number of terms to predict')
cmd:option('-only_emoji',1,'predict only emoji 0 = allow words')
cmd:option('-result_separator','\t','separator to use in output between each result')
cmd:option('-print_probs',false,'print probabilities with each result')
cmd:option('-gpuid',-1,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',0,'set to 0 to ONLY print the predicted text, no diagnostics')
cmd:option('-start_symbol','_START_','start symbol for vocab (should really be in the model')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- print parmas
gprint('Parameters:')
gprint(opt)

-- no GPU for now
-- gpu_utils.init()
torch.setdefaulttensortype('torch.FloatTensor')

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not opt.model then
    error('Error: No model file specified, provide one with -model')
end
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. ')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local word2class = checkpoint.word2class
local class2word = {}
for term,i in pairs(word2class) do class2word[i] = term end

-- batch_size of 1 for online queries
local batch_size = 1

-- put everything into one flattened parameters tensor
if checkpoint.opt.hsm == 0 then
    params, grad_params = model_utils.combine_all_parameters(protos.rnn)
else
    params, grad_params = model_utils.combine_all_parameters(protos.rnn, protos.criterion)
end
gprint('Total number of parameters in the model: ' .. params:nElement())

-- setup initial state sizes
init_state_sizes = {}
for i=1,checkpoint.opt.num_layers do
    if checkpoint.opt.model == 'scrnn' then
        table.insert(init_state_sizes, checkpoint.opt.context_size)
    end
    table.insert(init_state_sizes, checkpoint.opt.rnn_size)
    if checkpoint.opt.model == 'lstm' then
        table.insert(init_state_sizes, checkpoint.opt.rnn_size)
    end
end

-- ship the model to the GPU if desired
protos = gpu_utils.ship_table(protos)

function forward(x)
    local curr_seq_length = #x

    local init_state_local = {}
    for _, init_size in ipairs(init_state_sizes) do
        table.insert(init_state_local, torch.zeros(batch_size, init_size))
    end
    -- ship to gpu
    x = gpu_utils.ship(torch.FloatTensor{x})
    init_state_local = gpu_utils.ship_table(init_state_local)

    local prediction = {}
    local rnn_state = {[0] = init_state_local}
    -- forward pass
    for t=1, curr_seq_length do
        local lst = protos.rnn:forward{x[{{},t}], unpack(rnn_state[t-1])}
        -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
        rnn_state[t] = {}
        for i=1,#init_state_sizes do table.insert(rnn_state[t], lst[i]) end
        prediction = lst[#lst]
    end
    collectgarbage()
    return prediction
end

function lookup_words(tokens, lookup_table, oov_vec)
    local ids = {}
    for _,tok in ipairs(tokens) do
        local val = lookup_table[tok]
        if type(val) ~= nil then
            table.insert(ids, val)
        else
            val = lookup_table[tok:lower()]
            if type(val) ~= nil then
                table.insert(ids, val)
            else
                table.insert(ids, oov_vec)
            end
        end
    end
    return ids
end

function processInput(word2class, class2word)
    local sep = opt.result_separator
    -- first input class is OOV
    local oov = class2word[1]
    for line in io.lines() do
        local split_text = line:split('%s+')
        if #split_text > 0 then
            -- returns table of embedding Tensors
            local tokens = {opt.start_symbol}
            for _,t in ipairs(split_text) do table.insert(tokens, t) end
            local x = lookup_words(tokens, word2class, oov)
            local prediction = forward(x)
            if opt.print_probs then
                prediction = torch.exp(prediction):squeeze()
                -- renormalize so probs sum to one
                prediction:div(torch.sum(prediction))
            end
            local probs, ids = torch.sort(prediction, true)
            probs = probs[1]
            ids = ids[1]
            local npredictions = 0
            for i = 1,ids:size(1) do
                if npredictions == opt.npredictions then break end
                local accepted = false
                if 0 ~= opt.only_emoji then
                    accepted = emoji.isEmoji(class2word[ids[i]])
                else
                    accepted = ids[i] > 5
                end
                if accepted then
                    io.write(class2word[ids[i]])
                    if opt.print_probs then
                        io.write(' '..probs[i])
                    end
                    io.write(sep)
                    npredictions = npredictions + 1
                end
            end
            io.write('\n')
        end
    end
    return 0
end

return processInput(word2class, class2word)

