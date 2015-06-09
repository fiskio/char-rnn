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

local tblx = require 'pl.tablex'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-sample',false,'false to use max at each timestep, true to sample at each timestep')
cmd:option('-primetext'," ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-vocab','vocab.txt','vocabulary whitelist filter')

cmd:option('-stop_list',{'%S'} ,'regex of stop characters')
cmd:option('-branch',20,'number of next characters to consider at each step')
cmd:option('-prune',20,'maximum number of next characther expansion at each step')
cmd:option('-depth',10,'maxiumum length of any predicted words')
cmd:option('-n',10,'maxiumum length of any predicted words')

cmd:option('-debug',false,'print excessive debug')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

local dprint = function(str)
   if opt.debug then print(str) end
end

local surround = function(str)
   return '*'..str..'*'
end

if opt.gpuid >= 0 then
    print('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    print('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?') end
checkpoint = torch.load(opt.model)

-- load vocab filter (whitelist)
local vocab_filter = {}
for line in io.lines(opt.vocab) do
   local word = line:split('%s+')[2]
   if word then vocab_filter[word] = true end
end

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
print('seeding with: ' .. seed_text..'\n')
for c in seed_text:gmatch'.' do
    prev_char = torch.Tensor{vocab[c]}
    if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
    local embedding = protos.embed:forward(prev_char)
    current_state = protos.rnn:forward{embedding, unpack(current_state)}
    if type(current_state) ~= 'table' then current_state = {current_state} end
end

local hit, miss, wtot = 0, 0, 0
local words = {}

function expandCtx(ctx, n)
   local out = torch.Tensor(ctx:size(1)*n, ctx:size(2)):typeAs(ctx)
   local idx = 1
   for i=1,ctx:size(1) do
      for j=1,n do
         out[{{idx},{}}] = ctx[i]
         idx = idx + 1
      end
   end
   return out
end

-- @states    table of tensors, each row is a current partial words
-- @probs     array of probabilities for each row of prefixes tensors
-- @contexts  array of partial words for each row of prefixes tensors
local function branch_next(states, prefixes, probs, n_best)
   dprint(states)
   -- softmax from previous timestep
   local next_h = states[#states]
   next_h = next_h / opt.temperature
   local log_probs = protos.softmax:forward(next_h)

   -- get n_best possible expansions
   dprint(log_probs:size())
   local sorted_probs, sorted_ids = torch.sort(log_probs, 2, true)
   sorted_probs = sorted_probs:narrow(2, 1, n_best)
   sorted_ids = sorted_ids:narrow(2, 1, n_best)
   dprint(sorted_probs:size(), sorted_ids:size())

   -- create table of new expanded prefixes
   local new_prefixes = {}
   --print(states[1]:size(1), #prefixes, #probs, sorted_ids:size())
   --if #prefixes ~= sorted_ids:size(1) then print(prefixes) end
   for i, prefix in ipairs(prefixes) do
      for j=1,n_best do
         local char_id = sorted_ids[i][j]
         local np = prefix .. ivocab[char_id]
         table.insert(new_prefixes, np)
      end
   end
   -- create probabilities for new prefixes
   local new_probs = {}
   for i, prob in ipairs(probs) do
      for j=1,n_best do
         local ep = sorted_probs[i][j]
         local np = prob + ep
         table.insert(new_probs, np)
      end
   end

   -- forward the rnn for new states
   --print(sorted_ids)
   local foo = sorted_ids:reshape(sorted_ids:nElement())
   --print('FOO')
   dprint(foo)
   local embedding = protos.embed:forward(foo)
   --print(embedding)
   --print(states)

   -- forward the network for the next iteration
   for i,state in ipairs(states) do
      --states[i] = state:repeatTensor(n_best, 1)
      states[i] = expandCtx(state, n_best)
   end
   local new_states = protos.rnn:forward{embedding, unpack(states)}
   if type(new_states) ~= 'table' then new_states = {new_states} end

   return new_states, new_prefixes, new_probs
end

-- return the best n more probable over all
local function prune_bestOverAll(states, prefixes, probs, n_best)
   dprint(probs)
   -- find the most probables
   local tprobs = torch.Tensor(probs)
   local sorted_probs, sorted_ids = torch.sort(tprobs, true)
   sorted_probs = sorted_probs:sub(1, n_best)
   sorted_ids = sorted_ids:sub(1, n_best):long()
   local best_probs = sorted_probs:totable()
   dprint(sorted_ids, sorted_probs)

   -- select prefixes
   local best_prefixes = {}
   for i=1,sorted_ids:size(1) do
      local id = sorted_ids[i]
      table.insert(best_prefixes, prefixes[id])
   end

   -- select states
   dprint('best_prefixes', best_prefixes)

   dprint(states)
   for i, state in ipairs(states) do
      states[i] = state:index(1, sorted_ids)
   end
   dprint(states)
   return states, best_prefixes, best_probs
end

function extract_words(words, word_probs, states, prefixes, probs)
   --local words = {}
   --local word_probs= {}
   local word_ids = {}
   local keep_ids = {}
   -- is last charachter a stop?
   for i,pfx in ipairs(prefixes) do
      local last_char = pfx:sub(#pfx)
      if last_char:find('%A') then
         table.insert(words, pfx)
         table.insert(word_probs, probs[i])
         table.insert(word_ids, i)
      else
         table.insert(keep_ids, i)
      end
   end
   -- remove words from states, prefixes, probs
   local t_keep_ids = torch.LongTensor(keep_ids)
   for i, state in ipairs(states) do
      states[i] = state:index(1, t_keep_ids)
   end
   local new_prefixes, new_probs = {}, {}
   for _, kid in ipairs(keep_ids) do
      table.insert(new_prefixes, prefixes[kid])
      table.insert(new_probs, probs[kid])
   end
   assert(#prefixes == #probs)
   --assert(#prefixes == states[1][1]:size(1))
   return words, word_probs, states, new_prefixes, new_probs
end

-- aggregate the probability of multiple word+stop
-- i.e. 'am' = 'am ' + .. + 'am,'
function merge_words(words, probs)
   local wordToProb = {}
   for i, word in ipairs(words) do
      local last_char = word:sub(#word)
      local actual_word = word:sub(1,#word-1)
      local curr_prob = wordToProb[actual_word] or 0
      curr_prob = curr_prob + math.exp(probs[i])
      wordToProb[actual_word] = curr_prob
   end
   return wordToProb
end

function spairs(t, order)
    -- collect the keys
    local keys = {}
    for k in pairs(t) do keys[#keys+1] = k end

    -- if order function given, sort by it by passing the table and keys a, b,
    -- otherwise just sort the keys
    if order then
        table.sort(keys, function(a,b) return order(t, a, b) end)
    else
        table.sort(keys)
    end

    -- return the iterator function
    local i = 0
    return function()
        i = i + 1
        if keys[i] then
            return keys[i], t[keys[i]]
        end
    end
end

local words, word_probs = {}, {}

local t1 = torch.Timer()
local states, prefixes, probs = branch_next(current_state, {""}, {0}, opt.branch)
--print(states, prefixes, probs)
for i=1,opt.depth do
   -- explore a new character expansion
   states, prefixes, probs = branch_next(states, prefixes, probs, opt.branch)
   -- keep only the best ones
   states, prefixes, probs = prune_bestOverAll(states, prefixes, probs, opt.prune)
   -- are there any words already?
   words, word_probs, states, prefixes, probs = extract_words(words, word_probs, states, prefixes, probs)
   -- shall we stop?
   if #words >= opt.n or #prefixes == 0 then break end
end

local wordToProb = merge_words(words, word_probs)
local total_time = t1:time().real

-- pretty print
local desord = function(x,y) return x > y end
for word, prob in tblx.sortv(wordToProb, desord) do
   print(string.format('%s\t%.5f', word, prob))
end

print(string.format('\nTotal Time: %.f ms', total_time*1000))

-- Also print current unfinished partial words
---[[
print('')
for i, pfx in ipairs(prefixes) do
   print(pfx, probs[i])
end
--]]
