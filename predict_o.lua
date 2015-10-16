require 'nngraph'
--require 'cunn' 
local tablex = require 'pl.tablex'
local dyn = require 'util/dynamic_ngrams'

cmd = torch.CmdLine()
cmd:text()
cmd:text("A little utility to predict words from context")
cmd:text()
cmd:text('Example:')
cmd:text('$> th predict.lua checkpoint.exp.th7')
cmd:text()
cmd:text('Options:')
cmd:argument('exp', 'Checkpoint containing the model', 'string')
cmd:option('-n', 20, 'Number of top results to show', 'number')
cmd:option('-cuda', false, 'Use GPU for predictions')
cmd:option('-challenge', false, 'Enable "challenge mode" for piping LM challenge commands')
cmd:option('-unigrams', '', 'Provide a background unigrams file to fall back to (--challenge mode only)')
cmd:option('-tokenizer', 'tokenize-text', 'Which tokenizer shell program to use (defaulting to the SDK\'s tokenize-text')
cmd:option('-dynamic', false, 'If enabled, dynamic learning is turned on (using a replica of the SDK\'s strategy)')
cmd:option('-case-backoff', 0.1, 'Sets the prefix matching case backoff penalty')
cmd:text()
opt = cmd:parse(arg or {})

torch.setdefaulttensortype('torch.FloatTensor')
pred_cache = {}
ctx_cache = {}
-------------------- Utility --------------------

local function readUnigrams(f)
   local results = {}
   for line in f:lines() do
      local tokens = line:split("%s")
      if 2 <= #tokens then
         results[tokens[1]] = tonumber(tokens[2])
      end
   end
   f:close()
   return results
end

-- like tablex.sortv, but include keys with duplicated values (in arbitrary order)
local function sortv(t)
   local keys = tablex.keys(t)
   table.sort(keys, function (a,b) return t[a] < t[b] end)
   local i = 0
   return function ()
      i = i + 1
      if i <= #keys then return keys[i], t[keys[i]] end
   end
end

-------------------- State --------------------

-- model
local checkpoint = torch.load(opt.exp)
local model = checkpoint.protos
local wordToClass = checkpoint.word2class
local classToWord = {}
for w,c in pairs(wordToClass) do
   classToWord[c] = w
end

-- CUDA?
if opt.cuda then
   require 'cunn'
   model.rnn = model.rnn:cuda()
   torch.setdefaulttensortype('torch.CudaTensor')
end

-- unigrams
local unigrams = opt.unigrams~='' and readUnigrams(io.popen("gzip -cd "..opt.unigrams)) or {}
local unigram_total = 0
for __,n in pairs(unigrams) do
   unigram_total = unigram_total + n
end
local sortedUnigrams = tablex.keys(unigrams)
table.sort(sortedUnigrams,
           function (a, b)
              return (unigrams[b] or 0) < (unigrams[a] or 0)
           end)

-- dynamic
local dynamic_model = opt.dynamic and dyn:create()
local start_of_sequence = '^'

-------------------- Utility --------------------

-- build the input token index vector, left-padded with "<S>"
local function readContext(tokens)
   local input = torch.LongTensor(1,#tokens+1):fill(2) -- XXX should be prettier..
   for i=2, input:size(2) do
      local token = tokens[i-1]
      input[1][i] = wordToClass[token]
                 or wordToClass[token:lower()]
		 or wordToClass[token:upper()]
                 or 1
   end
   return input
end

local function split(input)
   local quotedInput = string.format("%q", input):gsub('([`$])', '\\%1')
   local p = io.popen('echo '..quotedInput..' | '..opt.tokenizer, 'r')
   local tokens = p:read():split("%s")
   p:close()
   return tokens
end

local function splitContextCurrentWord(input)
   local tokens = split(input)
   if #tokens == 0 or (0 < #input and input:sub(#input):find("%s")) then
      -- we're at the end of a term, everything is context
      return tokens, ""
   else
      -- we're in the middle of the last term
      local head = table.remove(tokens, #tokens)
      return tokens, head
   end
end

local function hash(tensor)
   if not tensor then return nil end
   local hash = ''
   tensor:apply(function(x) hash = hash..x end)
   --print(tensor, 'hash: '..hash)
   return hash
end

local function getInitialContext(input)
   -- is all ctx already in cache?
   local cached_pred = pred_cache[hash(input)]
   if cached_pred then
      return nil, cached_pred
   end
   return input, nil
end   

local function getCachedInit(input)
   if not input then return nil, nil end
   if input:size(2) < 2 then return input, nil end
   -- is all ctx minus final word in cache
   local minus_one = input:sub(1,-1,1,input:size(2)-1)
   local cached_ctx = ctx_cache[hash(minus_one)]
   if cached_ctx then
      return input:sub(1,-1,input:size(2)-1,-1):cuda(), cached_ctx
   end
   return input, nil
end

local function predict(input)
   local orig_input = input
   local preds
   local init_state = {torch.zeros(1, checkpoint.opt.rnn_size):cuda()}
   if checkpoint.opt.model == 'lstm' then
      table.insert(init_state, torch.zeros(1, checkpoint.opt.rnn_size))
   end
   local input, cached_init = getCachedInit(input)
   local rnn_state = cached_init or init_state
   --local input, preds = getInitialContext(orig_input)
   model.rnn:evaluate()
   if input then
      --print(input) 
      for i=1,input:size(2) do
         local lst = model.rnn:forward{input[{{}, i}], unpack(rnn_state)}
         preds = lst[#lst][1]
         rnn_state = {}
         for k=1,#init_state do table.insert(rnn_state, lst[k]) end
      end
      ctx_cache[hash(orig_input)] = rnn_state
      --pred_cache[hash(orig_input)] = preds
   end
   -- cache prediction
   return preds
end

-------------------- Ranking --------------------

local function scoreNeural(scores, candidate)
   local class = wordToClass[candidate]
   return class and math.exp(scores[class]) or 0
end

local function scoreDynamic(context, candidate)
   if opt.dynamic then
      local context = tablex.copy(context)
      table.insert(context, 1, start_of_sequence)
      return dyn:evaluate(dynamic_model, context, candidate)
   else return 0 end
end

local function scoreUnigram(candidate)
   return 1E-12 * (unigrams[candidate] or 0) / unigram_total
end

local function cmdRank(context, candidates)
   local contextTokens = split(context)
   local neuralScores = predict(readContext(contextTokens))

   -- score each candidate (max from each model)
   local scores = {}
   for __,candidate in ipairs(candidates) do
      scores[candidate] = -math.max(
         scoreNeural(neuralScores, candidate),
         scoreDynamic(contextTokens, candidate),
         scoreUnigram(candidate)
      )
   end

   -- sort & return
   local results = {}
   for word,__ in sortv(scores) do
      table.insert(results, word)
   end
   return results
end

-------------------- Prediction --------------------

local function predictNeural(context, matchScore)
   -- get whole vocab predictions
   local scores,classes = torch.sort(predict(readContext(context)), true)

   -- decode the top opt.n completions of 'prefix'
   local results = {}
   for n=1,classes:nElement() do
      if opt.n <= tablex.size(results) then break end
      local class = classes[n]
      if class ~= 1 then -- XXX _OOV_ from model
         local word = classToWord[class]
         local match = matchScore(word)
         if match then
            results[word] = match * math.exp(scores[n])
         end
      end
   end
   return results
end

local function predictDynamic(context, matchScore)
   if dynamic_model then
      local context = tablex.copy(context)
      table.insert(context, 1, start_of_sequence)
      local results = dyn:predict(dynamic_model, context, opt.n, matchScore)
      results[start_of_sequence] = undefined
      return results
   else
      return {}
   end
end

local function predictUnigram(matchScore)
   local results = {}
   for __,word in ipairs(sortedUnigrams) do
      if opt.n <= tablex.size(results) then break end
      local match = matchScore(word)
      if match then
         results[word] = match * scoreUnigram(word)
      end
   end
   return results
end

local function mergePredictions(prefixLength, lists)
   local merged = {}
   for __,list in ipairs(lists) do
      for word,prob in pairs(list) do
         merged[word] = math.min(merged[word] or 0, -prob)
      end
   end
   local results = {}
   for word,__ in sortv(merged) do
      if opt.n <= tablex.size(results) then break end
      table.insert(results, string.sub(word, prefixLength+1))
   end
   return results
end

local function cmdPredict(input)
   local context, prefix = splitContextCurrentWord(input)
   local function prefixMatch(word)
      -- strict prefix
      if #word <= #prefix then
         return nil
      -- case sensitive match
      elseif string.sub(word, 1, #prefix) == prefix then
         return 1.0
      -- case insensitive backoff
      elseif string.sub(word, 1, #prefix):lower() == prefix:lower() then
         return opt["case-backoff"]
      else return nil end
   end
   return mergePredictions(#prefix, {
         predictNeural(context, prefixMatch),
         predictDynamic(context, prefixMatch),
         predictUnigram(prefixMatch)
   })
end

-------------------- Training --------------------

local function cmdTrain(text)
   if dynamic_model then
      local tokens = split(text)
      table.insert(tokens, 1, start_of_sequence)
      dyn:train(dynamic_model, tokens)
   end
end

local function cmdClear()
   dynamic_model = opt.dynamic and dyn:create()
end


-------------------- Script --------------------

if opt.challenge then
   local n = 0
   while true do
      local line = io.read("*line")
      if not line then break end
      local parts = line:split("\t")
      local cmd = table.remove(parts, 1)

      if cmd == "predict" then
         io.stdout:write(table.concat(cmdPredict(parts[1] or ""), '\t')..'\n')
      elseif cmd == "rank" then
         local context = table.remove(parts, 1) or ""
         local candidates = parts
         io.stdout:write(table.concat(cmdRank(context, candidates), '\t')..'\n')
      elseif cmd == "train" then
         cmdTrain(parts[1] or "")
      elseif cmd == "clear" then
         cmdClear()
      else
         io.stderr:write("Unknown command "..line.."\n")
      end
      io.stdout:flush()
      n = n + 1
      if n % 100 == 0 then
         collectgarbage()
      end
   end

else
   print("# Enter a sequence of words to see what comes next...")
   -- interactive run loop
   while true do
      io.write("# context: ")
      local line = io.read("*line")
      if not line then break end

      -- read in tokens from the input, left-padded with "<S>"
      local tokens = line:split("%s+")
      local input = readContext(tokens)
      io.write("# input: ")
      for i=1,input:nElement() do
         io.write(classToWord[input[{1,i}]].." ")
      end
      io.write("...\n")

      -- feed forward through the model
      local output = predict(input)

      -- sort and print out top predictions
      local scores,classes = torch.sort(output, true)
      for i=1,math.min(opt.n, classes:nElement()) do
         print(string.format("    %-15s %.5f", classToWord[classes[i]], math.exp(scores[i])))
      end
      print()
   end
end
