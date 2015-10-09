------------------------------------------------------------------------------------
-- [[ sk.TextSampler ]]
-- TODO explain myself
-----------------------------------------------------------------------------------
require 'math'
require 'paths'

local TextSampler, parent = torch.class("TextSampler")
TextSampler.isTextSampler = true -- XXX ?

function TextSampler:__init(config)
    assert(torch.type(config) == 'table' and not config[1], "Constructor requires key-value arguments")
    local args, mode, context_size, max_batch_size, max_length, start_id = xlua.unpack(
        {config},
        'TextSampler', nil,
        {arg='mode', type='string', req=true, help='sampling mode <rnn|ff|emoji-ff>'},
        {arg='context_size', type='number', req=false, help='size of context window to sample (for applicable modes)'},
        {arg='max_batch_size',  type='number', req=false, help='maximum batch size'},
        {arg='max_length', type='number', req=false, help='maximum number of tokens in a line; should match Text object'},
        {arg='start_id', type='number', req=false, help='id of sentence start token'}
        )
    self._mode = mode
    self._context_size = context_size
    self._max_batch_size = max_batch_size
    self._max_length = max_length
    self._start_id = start_id
    assert(self._mode == 'rnn' or
           self._mode == 'ff' or
           self._mode == 'emoji-ff',
           'TextSampler: invalid mode '..mode..' supplied.')
    if mode == 'rnn' then
        --nothing
    elseif mode == 'ff' then
        local check = function(param)
            assert(param, 'ff sampling mode requires '..param..' tp be set')
        end
        check(self._context_size)
        check(self._max_batch_size)
        check(self._max_length)
        check(self._start_id)
        self._block_position = 0 -- set to 0 to trigger a fetch on first call to next_batch
        self._ff_block = torch.Tensor(self._max_batch_size, self._context_size + self._max_length):fill(self._start_id)
    elseif mode == 'ff' then
       -- nothing to do :)
    end
end

function TextSampler:next_batch(text, batch_list)
   -- sampling for RNNs doesn't need caching of batches, it's 1-1 with text's next_batch
   if self._mode == 'rnn' then
      local batch = text:next_batch(batch_list)
      local x = batch
      local y = torch.Tensor():typeAs(x):resizeAs(x)
      -- shifted copy
      y:sub(1,-1,1,-2):copy(x:sub(1,-1,2,-1))
      -- end of sentence
      y[{{},-1}] = text:end_id()
      return x, y
   elseif self._mode == 'ff' then
      local ff_block = self._ff_block
      -- check if current ff_block can still be iterated over
      local pos = self._block_position
      local seq_size = self._context_size + 1
      if 0 < pos and pos+seq_size <= ff_block:size(2) then
         local x = ff_block:sub(1,-1, pos,pos+seq_size-1)
         local y = ff_block:sub(1,-1, pos+seq_size,pos+seq_size)
         self._block_position = pos + 1
         return x, y:contiguous():view(-1)
      else
         -- otherwise grab a new block from text to iterate over
         local text_batch = text:next_batch(batch_list)
         local batch_cols = text_batch:size(2)-1 -- text returns blocks prepended with a seq_start column
         local batch_rows = text_batch:size(1)
         -- make sure ff_block is no bigger than text block, with context_size collumns to the left
         ff_block:resize(batch_rows, batch_cols+self._context_size)
         ff_block:sub(1,-1, 1,self._context_size):fill(self._start_id)
         -- copy values into block to store for each iter
         ff_block:sub(1,batch_rows, -batch_cols,-1):copy(text_batch:sub(1,-1, 2,-1))
         self._block_position = 1
         pos = self._block_position
         local x = ff_block:sub(1,-1, pos,pos+seq_size-1)
         local y = ff_block:sub(1,-1, pos+seq_size,pos+seq_size)
         self._block_position = pos + 1
         --TODO transpose ff_block so we can return row slices and avoid a copy
         return x, y:contiguous():view(-1)
      end
   elseif self._mode == 'emoji-ff' then
      local text_batch = text:next_batch(batch_list)
      local x = text_batch:sub(1,-1, 1,-2)
      local y = text_batch:sub(1,-1, -1,-1)
      --TODO transpose ff_block so we can return row slices and avoid a copy
      return x, y:contiguous():view(-1)
    else
        error("What just happened? Unkown sampling mode in TextSampler.")
    end
end

function TextSampler:reset_batch_pointer(text, batch_list)
    self._block_position = 0 -- 0 to trigger refetch on next_batch
    text:reset_batch_pointer(batch_list)
end

TextSampler.modeSwitch = {
    rnn = 'rnn',
    irnn = 'rnn',
    gru = 'rnn',
    lstm = 'rnn',
    scrnn = 'rnn',
    cbow = 'ff'
}
TextSampler.networkToMode = function(network_type)
    local m = TextSampler.modeSwitch[network_type]
    if not m then
        error('no TextSampler mode for network type \''..network_type)
    end
    return m
end

