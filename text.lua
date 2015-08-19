------------------------------------------------------------------------------------
-- [[ sk.Text ]]
-- A dataset representing a corpus. It is assumed that the data directory
-- provided is has three files for training, test, and validation data,
-- each of which containing a sentence of space-separated tokens on each line,
-- and a fourth file containing the unigram counts ordered by decreasing frequency.
-- All files can be plain text or in .gz format.
-- i.e. data/train.txt.gz
--      data/valid.txt
--      data/test.txt
--      data/1-grams.txt
-----------------------------------------------------------------------------------
require 'paths'

local Text, parent = torch.class("Text")
Text.isText = true

function Text:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
   local args, name, data_path, train_file, valid_file, test_file, unig_file, vocab_cover, vocab_size,
         max_length, max_reps, batch_size, oov_sym, start_sym, end_sym, wb_sym, pad_sym
      = xlua.unpack(
      {config},
      'Text', nil,
      -- paths
      {arg='name',        type='string', req=true,                 help='dataset name'},
      {arg='data_path',   type='string', default='data',           help='data directory'},
      {arg='train_file',  type='string', default='train.txt.gz',   help='training file'},
      {arg='valid_file',  type='string', default='valid.txt.gz',   help='validation file)'},
      {arg='test_file',   type='string', default='test.txt.gz',    help='test file'},
      {arg='unig_file',   type='string', default='1-grams.txt.gz', help='unigram frequencies file'},
      -- sizes
      {arg='vocab_cover', type='string', default='100%',           help='word coverage of vocabulary'},
      {arg='vocab_size',  type='string', default=10000,            help='number of words in vocabulary'},
      {arg='max_length',  type='number', default=100,              help='maximum number of tokens in a line'},
      {arg='max_reps',    type='number', default=math.huge,        help='maximum repeated tokens in a line'},
      {arg='batch_size',  type='number', default=1,                help='maximum batch size'},
      -- special tokens
      {arg='oov',         type='string', default='_OOV_',          help='out-of-vobabulary token'},
      {arg='start',       type='string', default='_START_',        help='sentence start token'},
      {arg='end',         type='string', default='_END_',          help='sentence end token'},
      {arg='wb',          type='string', default='_WB_',           help='word break token'},
      {arg='pad',         type='string', default='_PAD_',          help='padding token'}
   )
   -- paths
   self._name = name
   self._data_path = data_path
   self._train_file = self:get_path(train_file)
   self._valid_file = self:get_path(valid_file)
   self._test_file = self:get_path(test_file)
   self._unig_file = self:get_path(unig_file)
   -- sizes
   self._vocab_cover = vocab_cover
   self._orig_vocab_size = vocab_size
   self._vocab_size = vocab_size
   self._max_length = max_length
   self._max_reps = max_reps
   self._batch_size = batch_size
   -- special symbols
   self._oov_sym = oov_sym
   self._start_sym = start_sym
   self._end_sym = end_sym
   self._wb_sym = wb_sym
   self._pad_sym = pad_sym

   if not self:load_cache() then
      -- load all input files
      self:load_vocab()
      --self._train_set = self:load_text(self._train_file)
      --self._valid_set = self:load_text(self._valid_file)
      --self._test_set  = self:load_text(self._test_file)
      -- initialise batch lists and index
      self._train_batches = self:listify(self:load_text(self._train_file))
      self._valid_batches = self:listify(self:load_text(self._valid_file))
      self._test_batches  = self:listify(self:load_text(self._test_file))
      self:reset_batch_pointer()
      -- save a copy
      self:save()
   end
end

function Text:load_vocab()
   -- initalise maps
   local class2word, word2class = {},  {}
   table.insert(class2word, self._oov_sym)
   table.insert(class2word, self._start_sym)
   table.insert(class2word, self._end_sym)
   table.insert(class2word, self._wb_sym)
   table.insert(class2word, self._pad_sym)
   for i,w in ipairs(class2word) do
      word2class[w] = i
   end
   local OOV = word2class[self._oov_sym]
   local vocab_cover = self._vocab_cover
   local vocab_size = self._vocab_size
   local vocab_file = self._unig_file
   local counts = {0, 0, 0, 0, 0}

   -- determine the right vocab_size
   if vocab_cover ~= '100%' then
      -- percentage of coverage + 5 reserved
      -- TODO: fix coverage
      vocab_size = self:setCoveragePercentage(vocab_file, vocab_cover) + #class2word
   else
      -- exact number + 5 reserved
      vocab_size = vocab_size + #class2word
   end
   assert(vocab_size < 32767, 'ShortTensor')
   -- load vocabulary from all files
   local next_id = #class2word + 1
   local total_count = 0
   -- read all lines
   for line in self:fileLines(vocab_file) do
      if line ~= '' then
         local tokens = line:split("%s+")
         local count = tokens[1]
         local next_word = tokens[2]
         if word2class[next_word] then
            -- seen before -> update
            print('Warning! duplicated unigram: '..next_word)
            local prevId = word2class[next_word]
            counts[prevId] = counts[prevId] + count
         elseif next_id <= vocab_size then
            -- new word
            class2word[next_id] = next_word
            word2class[next_word] = next_id
            counts[next_id] = count
            next_id = next_id + 1
         else
            -- _OOV_
            counts[OOV] = counts[OOV] + count
         end
         total_count = total_count + count
      end
   end
   -- set state
   self._class2word = class2word
   self._word2class = word2class
   self._unig_probs = torch.Tensor(counts):div(total_count)
   self._vocab_size = #class2word
end

-- @arg path   input text file
function Text:load_text(path)
   local sentence_set = {}
   local skipped = 0 -- too repetitive
   local trunc = 0   -- too long
   local empty = 0   -- empty
   local n_lines = self:get_line_count(path)
   print(string.format('Loading %s lines of text from %s...', n_lines, path))
   xlua.progress(0, n_lines)
   local curr_line = 0
   for line in self:fileLines(path) do
      curr_line = curr_line + 1
      -- is it empty?
      if line ~= '' then
         local sentence = { self:start_id() } -- prepend start_id
         local tokens = line:split('%s+')
         if #tokens > self._max_length then trunc = trunc + 1 end
         -- build table of word classes
         local length = math.min(self._max_length, #tokens)
         for i=1, length do
            local word = tokens[i]
            local wc = self._word2class[word] or self:oov_id()
            table.insert(sentence, wc)
         end
         assert(sentence[1] == self:start_id())
         assert(#sentence >= 2)
         if self:ok_repeat(sentence) then
            -- convert it to tensor
            local t_sentence = torch.ShortTensor(sentence):reshape(1, #sentence)
            local batch_list = sentence_set[#sentence]
            if batch_list then
               local batch = batch_list[#batch_list]
               if batch:size(1) >= self._batch_size then
                  table.insert(batch_list, t_sentence)
               else
                  batch_list[#batch_list] = batch:cat(t_sentence, 1)
               end
            else
               sentence_set[#sentence] = { t_sentence }
            end
         else
            skipped = skipped + 1
         end
      else
         empty = empty + 1
      end
      if curr_line % 1e4 == 0 then xlua.progress(curr_line, n_lines) end
   end
   xlua.progress(n_lines, n_lines)
   -- report
   print(string.format('Lines Empty:     %s - %.1f%%', empty, 100*empty/n_lines))
   print(string.format('Lines Skipped:   %s - %.1f%%', skipped, 100*skipped/n_lines))
   print(string.format('Lines Truncated: %s - %.1f%%', trunc, 100*trunc/n_lines))
   collectgarbage()
   return sentence_set
end

-- sets the vocabulary size to a given percentual coverage
-- FIXME
function Text:setCoveragePercentage(percent)
   -- count all words
   local n_percent = tonumber(percent:sub(1,-2)) -- remove trailing %
   local total_tokens = 0
   local classes = {}
   for line in self:fileLines(self._unig_file) do
      local tokens = line:split("%s+")
      local word = tokens[1]
      if not self:is_special_token(word) then
         local count = tokens[2]
         classes[word] = true
         total_tokens = total_tokens + count
      end
   end
   local total_coverage_words = math.floor((total_tokens * n_percent) / 100)
   local remainders = {}
   for fname, weight in pairs(unigrams_fileset) do
      remainders[idx] = weight * total_coverage_words
   end
   -- add words until remainder is > 0
   local words = 0
   for line in self:fileLines(self._unig_file) do
      local tokens = line:split("%s+")
      local word = tokens[1]
      -- if not OOV decrease remainder
      if not self:is_special_token(word) then
         words = words + 1
         local count = tokens[2]
         remainder = remainder - count
      end
      -- got enough?
      if (remainder <= 0) then break end
   end
   print('Coverage ' .. n_percent .. '% of ' .. total_tokens .. ' tokens is: ' .. coverage_words)
   print('... which corresponds to the ' .. words .. ' most common words')
   collectgarbage()
   return words
end

-- flatten batch table into shuffled array
function Text:listify(sentence_set)
   local batch_list = {}
   for len, list in pairs(sentence_set) do
      for idx, batch in ipairs(list) do
         table.insert(batch_list, batch)
      end
   end
   return self:shuffle(batch_list)
end

-- shuffles elements of an array
function Text:shuffle(list)
   for i = #list, 2, -1 do
      local idx = math.random(i)
      list[i], list[idx] = list[idx], list[i]
   end
   return list
end

-- restart batch counter
function Text:reset_batch_pointer(batch_set)
   if batch_set then
      self._batch_index[batch_set] = 14300
   else
      self._batch_index = {}
      self._batch_index[self._train_batches] = 14300
      self._batch_index[self._valid_batches] = 14300
      self._batch_index[self._test_batches] = 14300
   end
end

-- get the next batch from a list
-- ie: txt:next_batch(txt:test_batches())
function Text:next_batch(batch_list)
   -- increment index
   self._batch_index[batch_list] = self._batch_index[batch_list] + 1
   if self._batch_index[batch_list] > #batch_list then
      self._batch_index[batch_list] = 1 end
   -- load inputs and targets
   local inputs = batch_list[self._batch_index[batch_list]]
   local targets = torch.Tensor():typeAs(x):resizeAs(x)
   -- shifted copy of inputs
   targets:sub(1,-1,1,-2):copy(inputs:sub(1,-1,2,-1))
   -- end of sentence
   targets[{{},-1}] = self:end_id()
   return inputs, targets
end

-- check for excessive repetition
function Text:ok_repeat(list)
   local last = ''
   local reps = 0
   for _,w in ipairs(list) do
      if w == last then
         reps = reps + 1
      else
         last = w
         reps = 0
      end
      if reps > self._max_reps then
         return false
      end
   end
   return true
end

-- concatenate the path
function Text:get_path(fname)
   local path = paths.concat(self._data_path, self._name, fname)
   assert(paths.filep(path), 'File not found! '..path)
   return path
end

-- transparently read lines from gzip and plain text files
function Text:fileLines(path)
   return io.popen('gunzip -c -f '..path, 'r'):lines()
end

-- check if a token is a reserved one
function Text:is_special_token(token)
   return token == self._oov_sym or
          token == self._start_sym or
          token == self._end_sym or
          token == self._wb_sym or
          token == self._pad_sym
end

-- fast way of counting lines in a file
function Text:get_line_count(file)
   local res = sys.execute('gunzip -c -f '..file..' | wc -l')
   local all = res:split('%s+')
   return tonumber(all[1])
end

-- save a copy
function Text:save()
   collectgarbage()
   local path = paths.concat(self._data_path, self._name, self._name..'.t7')
   torch.save(path, self)
end

-- attempt loading from cache
function Text:load_cache()
   local path = paths.concat(self._data_path, self._name, self._name..'.t7')
   if not paths.filep(path) then return false end
   local cached = torch.load(path)
   -- helper
   local function check_param(param)
      if cached[param] ~= self[param] then
         error(string.format('Load error! %s cache: %s, self: %s', param, cached[param], self[param]))
      end
      return true
   end
   check_param('_train_file')
   check_param('_valid_file')
   check_param('_test_file')
   check_param('_unig_file')
   check_param('_orig_vocab_size')
   check_param('_max_length')
   check_param('_max_reps')
   check_param('_batch_size')
   check_param('_oov_sym')
   check_param('_start_sym')
   check_param('_end_sym')
   check_param('_wb_sym')
   check_param('_pad_sym')
   -- all good
   print('Using compatible cached version: '..path)

   -- initialise batch lists and index
   self._vocab_size = cached._vocab_size
   self._train_batches = cached._train_batches
   self._valid_batches = cached._valid_batches
   self._test_batches  = cached._test_batches
   self._word2class = cached._word2class
   self._class2word = cached._class2word
   self._batch_index = cached._batch_index
   self:reset_batch_pointer()
   return true
end

-- [[ getters ]]
-- vocab maps
function Text:word2class() return self._word2class end
function Text:class2word() return self._class2word end

-- special classes
function Text:oov_id()   return self._word2class[self._oov_sym]   end
function Text:start_id() return self._word2class[self._start_sym] end
function Text:end_id()   return self._word2class[self._end_sym]   end
function Text:wb_id()    return self._word2class[self._wb_sym]    end
function Text:pad_id()   return self._word2class[self._pad_sym]   end
-- special symbols
function Text:oov_sym()   return self._oov_sym   end
function Text:start_sym() return self._start_sym end
function Text:end_sym()   return self._end_sym   end
function Text:wb_sym()    return self._wb_sym    end
function Text:pad_sym()   return self._pad_sym   end
-- batch lists
function Text:train_batches() return self._train_batches end
function Text:valid_batches() return self._valid_batches end
function Text:test_batches()  return self._test_batches  end
