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
cmd:option('-sample',false,'false to use max at each timestep, true to sample at each timestep')
cmd:option('-primetext'," ",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-length',2000,'number of characters to sample')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-vocab','vocab.txt','vocabulary whitelist filter')
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
print('seeding with ' .. seed_text)
for c in seed_text:gmatch'.' do
    prev_char = torch.Tensor{vocab[c]}
    if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
    local embedding = protos.embed:forward(prev_char)
    current_state = protos.rnn:forward{embedding, unpack(current_state)}
    if type(current_state) ~= 'table' then current_state = {current_state} end
end

local hit, miss, wtot = 0, 0, 0
local words = {}

-- @states    table of tensors, each row is a current partial words
-- @probs     array of probabilities for each row of prefixes tensors
-- @contexts  array of partial words for each row of prefixes tensors
local function branch_next(states, prefixes, probs)
   print(states)
   -- softmax from previous timestep
   local next_h = states[#states]
   next_h = next_h / opt.temperature
   local log_probs = protos.softmax:forward(next_h)

   -- get n_best possible expansions
   print(log_probs:size())
   local n_best = 5
   local sorted_probs, sorted_ids = torch.sort(log_probs, 2, true)
   sorted_probs = sorted_probs:narrow(2, 1, n_best)
   sorted_ids = sorted_ids:narrow(2, 1, n_best)
   print(sorted_probs:size(), sorted_ids:size())

   -- create table of new expanded prefixes
   local new_prefixes = {}
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
   print(sorted_ids)
   local foo = sorted_ids[1]
   print(foo)
   local embedding = protos.embed:forward(foo)
   print(embedding)
   local new_states = protos.rnn:forward{embedding, unpack(states)}
   if type(new_states) ~= 'table' then new_states = {new_states} end

   return new_states, new_prefixes, new_prefixes
end
branch_next(current_state, {""}, {0})
error()
-- onpredict next char
local function next_char(state, spi)
   -- softmax from previous timestep
   local next_h = state[spi]
   next_h = next_h / opt.temperature
   local log_probs = protos.softmax:forward(next_h)
   print(next_h:size())

   if opt.sample then
      -- use sampling
      local probs = torch.exp(log_probs):squeeze()
      prev_char = torch.multinomial(probs:float(), 1):resize(1):float()
   else
      -- use argmax
      local _, prev_char_ = log_probs:max(2)
      prev_char = prev_char_:resize(1)
  end

   -- forward the rnn for next character
   local embedding = protos.embed:forward(prev_char)
   state = protos.rnn:forward{embedding, unpack(state)}
   if type(state) ~= 'table' then state = {state} end
   return prev_char[1], state
end

local function next_words(state, spi)
   local nc, nw
   local orig_state = state
   while true do
      if nc then
        if nc:find('%A') then
           break
        end
      end
      nc, state = next_char(state, spi)
      nc = ivocab[nc]
      if nc:find('%a') then
         nw = nw and nw..nc or nc
      end
   end
   -- is this a real word?
   if nw  and not vocab_filter[nw] then
        miss = miss + 1
        io.write(sys.COLORS.red..nw..'\027[00m ')
        return nil, orig_state
   end
   return nw, state
end

-- start sampling/argmaxing
while hit+miss <= opt.length do
   local next_word
   local next_state = current_state
   while true do
      next_word, next_state = next_words(next_state, state_predict_index)
      if next_word then break end
   end
   if next_word then
      if not words[next_word] then
         wtot = wtot + 1
         words[next_word] = true
      end
   end
   hit = hit + 1
   io.write(next_word..' ')
end
io.write('\n') io.flush()
print(string.format('Hit: %s Miss: %s or %.2f%%', hit, miss, 100 * (miss/hit)))
print(string.format('The model predicted %d different words, out of %d', wtot, hit+miss))