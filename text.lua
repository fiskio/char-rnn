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

Text._unknown_word   = 1  --  '_OOV_'
Text._sentence_start = 2  --  '^'
Text._sentence_end   = 3  --  '_END_'
Text._word_break     = 4  --  '→'
Text._padding        = 5  --  '_PAD_'
Text._unknown_word_symbol   = '_OOV_'
Text._sentence_start_symbol = '^'
Text._sentence_end_symbol   = '_END_'
Text._word_break_symbol     = '→'
Text._padding_symbol        = '_PAD_'

function Text:__init(config)
   config = config or {}
   assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
   local args, name, data_path, train_file, valid_file, test_file, unig_file, vocab_cover, vocab_size, max_length, max_reps, batch_size
      = xlua.unpack(
      {config},
      'Text', nil,
      {arg='name',        type='string', req=true,                 help='dataset name'},
      {arg='data_path',   type='string', default='data',           help='data directory'},
      {arg='train_file',  type='string', default='train.txt.gz',   help='training file'},
      {arg='valid_file',  type='string', default='valid.txt.gz',   help='validation file)'},
      {arg='test_file',   type='string', default='test.txt.gz',    help='test file'},
      {arg='unig_file',   type='string', default='1-grams.txt.gz', help='unigram frequencies file'},
      {arg='vocab_cover', type='string', default='100%',           help='word coverage of vocabulary'},
      {arg='vocab_size',  type='string', default=10000,            help='number of words in vocabulary'},
      {arg='max_length',  type='number', default=100,              help='maximum number of tokens in a line'},
      {arg='max_reps',    type='number', default=math.huge,        help='maximum number of repeated tokens in a line'},
      {arg='batch_size',  type='number', default=1,                help='maximum batch size'}
   )
   self._name = name
   self._data_path = data_path
   self._train_file = self:get_path(train_file)
   self._valid_file = self:get_path(valid_file)
   self._test_file = self:get_path(test_file)
   self._unig_file = self:get_path(unig_file)
   self._vocab_cover = vocab_cover
   self._orig_vocab_size = vocab_size
   self._vocab_size = vocab_size
   self._max_length = max_length
   self._max_reps = max_reps
   self._batch_size = batch_size

   if not self:load_cache() then
      -- load all input files
      self:load_vocab()
      self._train_set = self:load_text(self._train_file)
      self._valid_set = self:load_text(self._valid_file)
      self._test_set  = self:load_text(self._test_file)
      -- initialise batch lists and index
      self:reset_batch_pointer()
      self._train_batches = self:listify(self._train_set)
      self._valid_batches = self:listify(self._valid_set)
      self._test_batches  = self:listify(self._test_set)
      -- save a copy
      self:save()
   end
end

function Text:load_vocab()
   local OOV = Text._unknown_word
   local vocab_cover = self._vocab_cover
   local vocab_size = self._vocab_size
   local vocab_file = self._unig_file
   -- first elements are reserved
   local word2class = {
      [Text._unknown_word_symbol]   = Text._unknown_word,
      [Text._sentence_start_symbol] = Text._sentence_start,
      [Text._sentence_end_symbol]   = Text._sentence_end,
      [Text._word_break_symbol]     = Text._word_break,
      [Text._padding_symbol]        = Text._padding
   }
   local class2word = {
      Text._unknown_word_symbol,
      Text._sentence_start_symbol,
      Text._sentence_end_symbol,
      Text._word_break_symbol,
      Text._padding_symbol
   }
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
         local sentence = {}
         local tokens = line:split('%s+')
         if #tokens > self._max_length then trunc = trunc + 1 end
         -- build table of word classes
         local length = math.min(self._max_length, #tokens)
         for i=1, length do
            local word = tokens[i]
            local wc = self._word2class[word] or Text._unknown_word
            sentence[i] = wc
         end
         if self:ok_repeat(sentence) then
            -- convert it to tensor
            local t_sentence = torch.ShortTensor(sentence):reshape(1, length)
            local batch_list = sentence_set[length]
            if batch_list then
               local batch = batch_list[#batch_list]
               if batch:size(1) >= self._batch_size then
                  table.insert(batch_list, t_sentence)
               else
                  batch_list[#batch_list] = batch:cat(t_sentence, 1)
               end
            else
               sentence_set[length] = { t_sentence }
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
      if word ~= Text._sentence_start_symbol then
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
function Text:reset_batch_pointer()
   self._batch_index = 1
end

-- get the next batch from a list
-- ie: txt:next_batch(txt:test_batches())
function Text:next_batch(batch_list)
   self._batch_index = self._batch_index + 1
   if self._batch_index > #batch_list then self._batch_index = 1 end
   local x = batch_list[self._batch_index]
   local y = torch.Tensor():typeAs(x):resizeAs(x)
   -- shifted copy
   y:sub(1,-1,1,-2):copy(x:sub(1,-1,2,-1))
   -- end of sentence
   y[{{},-1}] = Text._sentence_end
   return x, y
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
   return token == Text._unknown_word_symbol or
          token == Text._sentence_start_symbol or
          token == Text._sentence_end_symbol or
          token == Text._word_break_symbol or
          token == Text._padding_symbol
end

-- fast way of counting lines in a file
function Text:get_line_count(file)
   local res = sys.execute('gunzip -c -f '..file..' | wc -l')
   local all = res:split('%s+')
   return tonumber(all[1])
end

-- save a copy
function Text:save()
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
   -- all good
   print('Using compatible cached version: '..path)

   -- initialise batch lists and index
   self:reset_batch_pointer()
   self._vocab_size = cached._vocab_size
   self._train_batches = cached._train_batches
   self._valid_batches = cached._valid_batches
   self._test_batches  = cached._test_batches
   self._word2class = cached._word2class
   self._class2word = cached._class2word
   return true
end

-- alias
function Text:train_batches() return self._train_batches end
function Text:valid_batches() return self._valid_batches end
function Text:test_batches()  return self._test_batches end
