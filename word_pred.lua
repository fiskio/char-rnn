--[[
-
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
local strx = require 'pl.stringx'
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
cmd:option('-pad','\n⇶⇉→❀','padding text to be added on the left of primetext')
cmd:option('-primetext',"",'used as a prompt to "seed" the state of the LSTM using a given sequence, before we sample.')
cmd:option('-temperature',1,'temperature of sampling')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-vocab','','vocabulary whitelist filter')

cmd:option('-pc',0.8,'minimum probility mass to consider at each branching step')
cmd:option('-min_branch',1,'minimum number of next characters to consider at each step')
cmd:option('-queue_size',20,'maximum number of elements in the queue at each step')
cmd:option('-depth',10,'maximum length of any predicted words')
cmd:option('-n',10,'number of ranked word candidates to return')

cmd:option('-normalise',false,'divide the word score by its length')
cmd:option('-verbose',false,'print excessive statistics')
cmd:option('-debug',false,'print excessive debug')
cmd:option('-lmc',false,'accepts LMChallenge MK-Ultra commands')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

--[[ Helpers ]]--
local dprint = function(str)
   if opt.debug then print(str) end
end

local surround = function(str)
   return '*'..str..'*'
end

local stderr = function(str)
   io.stderr:write(str..'\n')
end

--[[ Initialisation ]]--
if opt.gpuid >= 0 then
    stderr('using CUDA on GPU ' .. opt.gpuid .. '...')
    require 'cutorch'
    require 'cunn'
    cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
end
torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    stderr('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)

-- was a vocabulary whitelist given?
local vocab_filter, prefix_filter = {}, {}
local rej_words, rej_prefixes = {}, {}

if opt.vocab ~= '' then
   -- create word filter
   for line in io.lines(opt.vocab) do
      local word = line:split('%s+')[2]
      if word then vocab_filter[word] = true end
   end
   stderr('#vocab_filter:', tblx.size(vocab_filter))
   -- create prefix filter
   for w,_ in pairs(vocab_filter) do
      for i=1,#w-1 do
         local pfx = w:sub(1, i)
         if pfx then prefix_filter[pfx] = true end
      end
   end
   stderr('#prefix_filter:', tblx.size(prefix_filter))
end

local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- what characters are not letters?
local non_letter_chars = {}
local letter_chars = {}
for i,c in pairs(ivocab) do
   if c:find('%A') then
      table.insert(non_letter_chars, c)
   else
      table.insert(letter_chars, c)
   end
end
if opt.verbose then
   print('non_letter_chars: '..#non_letter_chars..' out of '..#ivocab)
   for i,c in ipairs(non_letter_chars) do
      print(i, surround(c))
   end
   print('')
   print('letter_chars: '..#letter_chars..' out of '..#ivocab)
   for i,c in ipairs(letter_chars) do
      print(i, surround(c))
   end
   print('')
end

--[[ Functions ]]--

-- reset the state of a given model
function init_model(checkpoint)
   --checkpoint = torch.load(opt.model)
   local protos = checkpoint.protos

   local model = checkpoint.opt.model
   stderr('resetting '..model:upper()..'...')
   local num_layers = checkpoint.opt.num_layers or 1 -- or 1 is for backward compatibility
   local states = {}
   for L=1,checkpoint.opt.num_layers do
      -- c and h for all layers
      local h_init = torch.zeros(1, checkpoint.opt.rnn_size)
      if opt.gpuid >= 0 then h_init = h_init:cuda() end
      table.insert(states, h_init:clone())
      if model == 'lstm' then
         table.insert(states, h_init:clone())
      end
   end
   protos.rnn:evaluate() -- put in eval mode so that dropout works properly
   return protos, states
end

-- seed model with given text characters
function seed(seed_text, model, states)
   -- left padding?
   if opt.pad ~= '' then seed_text = opt.pad..seed_text end
   stderr('seeding with: '..sys.COLORS.green..seed_text..'\027[00m ')
   local softmax
   for c in seed_text:gmatch'.' do
      prev_char = torch.Tensor{vocab[c]}
      if opt.gpuid >= 0 then prev_char = prev_char:cuda() end
      local lst = model.rnn:forward{prev_char, unpack(states)}
      -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
      local states_size = #states
      states = {}
      for i=1,states_size do table.insert(states, lst[i]) end
      softmax = lst[#lst] -- last element holds the log probabilities
   end
   return model, states, softmax
end

-- expand tensor
-- i.e. A x2 = A
--      B      A
--             B
--             B
function expand_tensor(tensor, n)
   local r = tensor:size(1) * n
   local c = tensor:size(2)
   return tensor:repeatTensor(1, n):reshape(r, c)
end

-- find the minimum number of columns to consider in a sorted
-- multi-softmax to cover a given probability threshold
-- i.e. softmax = .6 .3 .1 , min_cov .7 -> 2 as  (.6+.5+.3+.3)/2 = 0.85 > 0.7
--                .5 .3 .2                   but (.6+.5)/2 = 0.55 < 0.7
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

-- find the n_best expansion for each context given
local function branch_next(model, states, softmax, prefixes, probs, n_best)
   -- softmax from previous timestep
   local log_probs = softmax:div(opt.temperature)

   -- get n_best possible expansions
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
   local foo = sorted_ids:reshape(sorted_ids:nElement())

   -- forward the network for the next iteration
   for i,state in ipairs(states) do
      --states[i] = expandTensor(state, n_best)
      states[i] = expand_tensor(state, n_best)
   end
   -- forward the rnn for next character
   local lst = model.rnn:forward{foo, unpack(states)}
   local new_states = {}
   for i=1,#states do table.insert(new_states, lst[i]) end
   local softmax = lst[#lst] -- last element holds the log probabilities

   return new_states, softmax, new_prefixes, new_probs
end

-- removes all non-acceptable partial words
local function prune_by_prefix(states, softmax, prefixes, probs)
   if opt.vocab == '' then
      return states, softmax, prefixes, probs
   end
   local new_prefixes, new_probs, keep_states = {}, {}, {}
   -- is it an allowed prefix?
   for i, pfx in ipairs(prefixes) do
      local last_char = pfx:sub(#pfx)
      if prefix_filter[pfx] or vocab_filter[pfx] or last_char:find('%A') then
         table.insert(new_prefixes, pfx)
         table.insert(new_probs, probs[i])
         table.insert(keep_states, i)
      else
         local rej_count = rej_prefixes[pfx] or 0
         rej_prefixes[pfx] = rej_count + 1
      end
   end
   -- prune states
   local t_keep_states = torch.LongTensor(keep_states)
   for i, state in ipairs(states) do
      states[i] = state:index(1, t_keep_states)
   end
   -- prune sofmax
   softmax = softmax:index(1, t_keep_states)

   return states, softmax, new_prefixes, new_probs
end

-- return the best n more probable over all
-- this is equivalend to bounding the priority queue
local function prune_bestOverAll(states, softmax, prefixes, probs, n_best)
   -- find the most probables
   local tprobs = torch.Tensor(probs)
   local sorted_probs, sorted_ids = torch.sort(tprobs, true)
   if n_best < sorted_probs:size(1) then
      sorted_probs = sorted_probs:sub(1, n_best)
      sorted_ids = sorted_ids:sub(1, n_best):long()
   end
   local best_probs = sorted_probs:totable()

   -- select prefixes
   local best_prefixes = {}
   for i=1,sorted_ids:size(1) do
      local id = sorted_ids[i]
      table.insert(best_prefixes, prefixes[id])
   end
   dprint('best_prefixes', best_prefixes)

   -- select states
   for i, state in ipairs(states) do
      states[i] = state:index(1, sorted_ids)
   end
   dprint(states)
   -- select softmax
   softmax = softmax:index(1, sorted_ids)

   return states, softmax, best_prefixes, best_probs
end

-- removes all complete words
function extract_words(words, word_probs, states, softmax, prefixes, probs)
   local word_ids = {}
   local keep_ids = {}
   -- is last charachter a stop?
   for i,pfx in ipairs(prefixes) do
      local last_char = pfx:sub(#pfx)
      if last_char:find('%A') then
         -- check if it is an allowed word
         local word = pfx:sub(1, #pfx-1)
         if opt.vocab == '' or vocab_filter[word] then
            table.insert(words, pfx)
            table.insert(word_probs, probs[i])
            table.insert(word_ids, i)
         else
            local rej_count = rej_words[word] or 0
            rej_words[word] = rej_count + 1
         end
      else
         table.insert(keep_ids, i)
      end
   end
   -- remove words from states, prefixes, probs, softmax
   if #keep_ids > 0 then
      local t_keep_ids = torch.LongTensor(keep_ids)
      softmax = softmax:index(1, t_keep_ids)
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
   return words, word_probs, states, softmax, new_prefixes, new_probs
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

-- divide the log probability of a word by its length
function normalise(wordToProb)
   for w, p in pairs(wordToProb) do
      wordToProb[w] = p / w:len()
   end
   return wordToProb
end

-- find the partial word, if any, in the seeding context
function extract_partial_word(context)
   local p_word = ''
   for i=#context,1,-1 do
      local c = context:sub(i,i)
      if c:find('%A') then break end
      p_word = c .. p_word
   end
   stderr('partial_word: ', surround(p_word))
   return p_word
end

-- predict the next most likely words
function predict_words(context, model, states, softmax)
   local words, word_probs = {}, {}
   local prefixes = {extract_partial_word(context)}
   local probs = {0}
   local wordToProb = {}
   local t1 = torch.Timer()

   -- forward a few times
   for i=1, opt.depth do
      -- explore a new character expansion
      states, softmax, prefixes, probs = branch_next(model, states, softmax, prefixes, probs, opt.min_branch)
      -- filter bogous sub-words
      states, softmax, prefixes, probs = prune_by_prefix(states, softmax, prefixes, probs)
      -- keep only the best ones
      states, softmax, prefixes, probs = prune_bestOverAll(states, softmax, prefixes, probs, opt.queue_size)
      -- are there any words already?
      words, word_probs, states, softmax, prefixes, probs = extract_words(words, word_probs, states, softmax, prefixes, probs)
      -- merge probabilities
      wordToProb = merge_words(wordToProb, words, word_probs)
      -- shall we stop?
      if #prefixes == 0 then break end
   end

   -- normalise probabilities by word length
   if opt.normalise then
      wordToProb = normalise(wordToProb)
   end

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

   -- show time
   stderr(string.format('\nTotal Time: %.f ms', 1000*t1:time().real))

   -- STATS?
   if opt.verbose then print_stats(rest, prefixes) end

   return topN
end

-- verbose statistics
function print_stats(rest, prefixes)
   -- Also print second best
   print('\nWords found, but not in topK -- i.e. second best')
   for word, prob in tblx.sortv(rest, desord) do
      print(string.format('%.6f\t%s', prob, word))
   end

   -- Also print current unfinished partial words
   print('\nPartial words still to explore')
   for i, pfx in ipairs(prefixes) do
      print(string.format('%s', pfx..'..'))
   end

   -- Also print rejected words
   print('\nRejected words')
   for word, count in pairs(rej_words) do
      print(count, sys.COLORS.red..word)
   end

   -- Also print rejected sub-words
   print('\nRejected sub-words')
   for pfx, count in pairs(rej_prefixes) do
      print(count, sys.COLORS.yellow..pfx)
   end
end

--[[ LMC ]]--

-- predict the next most likely words
function lmc_predict(tokens)
   assert(#tokens == 2)
   collectgarbage()
   local context = tokens[2]
   local model, states = init_model(checkpoint)
   local softmax
   model, states, softmax = seed(context, model, states)
   local predictions = predict_words(context, model, states, softmax)
   local skip_chars = 1+#(extract_partial_word(context))
   local output = ''
   for word, prob in tblx.sortv(predictions, desord) do
      output = output..word:sub(skip_chars)..'\t'
   end
   io.write(output..'\n')
   io.flush()
end

-- give a likelihood score for a given word
function score_word(model, states, word)
   local w_score = 0
   for c in word:gmatch'.' do
      -- score softmax
      local next_h = states[#states]
      next_h = next_h / opt.temperature
      local log_probs = model.softmax:forward(next_h)
      -- add probability of current character
      local char_id = vocab[c]
      w_score = w_score + log_probs[1][char_id]
      -- forward the network
      local embedding = model.embed:forward(torch.Tensor{char_id})
      states = model.rnn:forward{embedding, unpack(states)}
      if type(new_states) ~= 'table' then new_states = {new_states} end
   end
   w_score = math.exp(w_score)
   -- add score for terminating chars
   local next_h = states[#states]
   next_h = next_h / opt.temperature
   local log_probs = model.softmax:forward(next_h)
   local nlc_score = 0
   for i,c in ipairs(non_letter_chars) do
      nlc_score = math.exp(nlc_score + log_probs[1][i])
   end
   local w_score = w_score + nlc_score
   if opt.normalise then
      local p_map = normalise({[word] = score})
      w_score = p_map[word]
   end
   return w_score
end

-- rank a number of given words by likelihood
function lmc_rank(tokens)
   local context = tokens[2]
   local wordToProb = {}
   for i=3,#tokens do
      local model, states = init_model(checkpoint)
      local softmax
      model, states, softmax = seed(context, model, states)
      local word = tokens[i]
      wordToProb[word] = score_word(model, states, word, softmax)
   end
   local output = ''
   for word, prob in tblx.sortv(wordToProb, desord) do
      output = output..word..'\t'
   end
   print(output)
end

function lmc_eval(tokens)
   error('unimplemented')
end

function lmc_train()
   -- noop
end

--[[ MAIN ]]--
if opt.lmc then
   -- keep on accepting commands from stdin
   while true do
      local line = io.read('*line')
      if not line then break end
      local tokens = strx.split(line, '\t') -- tab separated
      -- which command?
      local cmd = tokens[1]
      if cmd == 'predict' then
         lmc_predict(tokens)
      elseif cmd == 'rank' then
         lmc_rank(tokens)
      elseif cmd == 'eval' then
         lmc_eval(tokens)
      elseif cmd == 'train' then
         lmc_train()
      else
         error('Unrecognised LMC command: '..cmd)
      end
   end
else
   -- just predict the next most likely words
   local model, states = init_model(checkpoint)
   local softmax
   if opt.primetext ~= '' then
      model, states, softmax = seed(opt.primetext, model, states)
   end
   local predictions = predict_words(opt.primetext, model, states, softmax)
   for word, prob in tblx.sortv(predictions, desord) do
      print(string.format('%.6f\t%s', prob, word))
   end
end