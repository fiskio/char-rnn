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
local desord = function(x,y) return x > y end

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-primetext'," ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-vocab','vocab.txt','vocabulary whitelist filter')

cmd:option('-pc',0.8,'minimum probility mass to consider at each branching step')
cmd:option('-min_branch',1,'minimum number of next characters to consider at each step')
cmd:option('-queue_size',20,'maximum number of elements in the queue at each step')
cmd:option('-depth',10,'maximum length of any predicted words')
cmd:option('-n',10,'number of ranked word candidates to return')

cmd:option('-verbose',false,'print excessive statistics')
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
local states, state_predict_index
local model = checkpoint.opt.model

print('creating a '..model:upper()..'...')
local num_layers = checkpoint.opt.num_layers or 1 -- or 1 is for backward compatibility
states = {}
for L=1,checkpoint.opt.num_layers do
    -- c and h for all layers
    local h_init = torch.zeros(1, opt.rnn_size)
    if opt.gpuid >= 0 then h_init = h_init:cuda() end
    table.insert(states, h_init:clone())
    if model == 'lstm' then
       table.insert(states, h_init:clone())
    end
end
state_predict_index = #states -- last one is the top h
local seed_text = opt.primetext
local prev_char

protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- do a few seeded timesteps
print('seeding with: ' .. seed_text..'\n')
for c in seed_text:gmatch'.' do
    prev_char = torch.Tensor{vocab[c]}
    if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
    local embedding = protos.embed:forward(prev_char)
    states = protos.rnn:forward{embedding, unpack(states)}
    if type(states) ~= 'table' then states = {states} end
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

function branch_coverage(probs, min_cov)
   local n_branches = 0
   local cov = 0
   for i=1, probs:size(2) do
      -- this is really cool!
      cov = torch.exp(probs[{{}, {1, i}}]):sum() / probs:size(1)
      n_branches = i
      if cov >= min_cov then break end
   end
   return n_branches, cov
end

-- @states    table of tensors, each row is a current partial words
-- @probs     array of probabilities for each row of prefixes tensors
-- @contexts  array of partial words for each row of prefixes tensors
local function branch_next(states, prefixes, probs, n_best)
   prefixes = prefixes or {''}
   probs = probs or {0}
   dprint(states)
   -- softmax from previous timestep
   local next_h = states[#states]
   next_h = next_h / opt.temperature
   local log_probs = protos.softmax:forward(next_h)

   -- get n_best possible expansions
   dprint(log_probs:size())
   local sorted_probs, sorted_ids = torch.sort(log_probs, 2, true)
   local n_branches, converage = branch_coverage(sorted_probs, opt.pc)
   local n_best = (n_branches < opt.min_branch) and opt.min_branch or n_branches
   -- show branching stats
   if opt.verbose then
      print(opt.min_branch, n_branches, converage, n_best)
   end

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
   --n_best = (n_best < sorted_probs:size(1)) and n_best or sorted_probs:size(1)
   if n_best < sorted_probs:size(1) then
      sorted_probs = sorted_probs:sub(1, n_best)
      sorted_ids = sorted_ids:sub(1, n_best):long()
   end
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
   if #keep_ids > 0 then
      local t_keep_ids = torch.LongTensor(keep_ids)
      for i, state in ipairs(states) do
         states[i] = state:index(1, t_keep_ids)
      end
   end
   local new_prefixes, new_probs = {}, {}
   for _, kid in ipairs(keep_ids) do
      table.insert(new_prefixes, prefixes[kid])
      table.insert(new_probs, probs[kid])
   end
   assert(#prefixes == #probs)
   return words, word_probs, states, new_prefixes, new_probs
end

-- aggregate the probability of multiple word+stop
-- i.e. 'am' = 'am ' + .. + 'am,'
function merge_words(wordToProb, words, probs)
   wordToProb = wordToProb or {}
   words = words or {}
   for i, word in ipairs(words) do
      local last_char = word:sub(#word)
      local actual_word = word:sub(1,#word-1)
      local curr_prob = wordToProb[actual_word] or 0
      curr_prob = curr_prob + math.exp(probs[i])
      wordToProb[actual_word] = curr_prob
   end
   return wordToProb
end

function normalise(wordToProb)
   for w, p in ipairs(wordToProb) do
      wordToProb[w] = p / w:len()
   end
   return wordToProb
end

local words, word_probs = {}, {}

local t1 = torch.Timer()

local prefixes, prob, wordToProb
-- forward a few times
for i=1, opt.depth do
   -- explore a new character expansion
   states, prefixes, probs = branch_next(states, prefixes, probs, opt.min_branch)
   -- keep only the best ones
   states, prefixes, probs = prune_bestOverAll(states, prefixes, probs, opt.queue_size)
   -- are there any words already?
   words, word_probs, states, prefixes, probs = extract_words(words, word_probs, states, prefixes, probs)
   -- merge probabilities
   wordToProb = merge_words(wordToProb, words, word_probs)
   -- shall we stop?
   if #prefixes == 0 then break end
end

-- normalise probabilities by word length
wordToProb = normalise(wordToProb)

local topN, rest = {}, {}
local size = 0
for word, prob in tblx.sortv(wordToProb, desord) do
   if size < opt.n then
      size = size + 1
      topN[word] = prob
   else
      rest[word] = prob
   end
end

local total_time = t1:time().real

-- pretty print
local desord = function(x,y) return x > y end
for word, prob in tblx.sortv(topN, desord) do
   print(string.format('%.6f\t%s', prob, word))
end

print(string.format('\nTotal Time: %.f ms', total_time*1000))

if opt.verbose then
   -- Also print second best
   print('')
   for word, prob in tblx.sortv(rest, desord) do
      print(string.format('%.6f\t%s', prob, word))
   end

   -- Also print current unfinished partial words
   print('')
   for i, pfx in ipairs(prefixes) do
      print(string.format('%.6f\t%s', math.exp(probs[i]), pfx..'..'))
   end
end
