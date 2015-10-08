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
    local args, mode, context_size = xlua.unpack(
        {config},
        'TextSampler', nil,
        {arg='mode', type='string', req=true, help='sampling mode <rnn|ff>'},
        {arg='context_size', type='number', req=false, help='size of context window to sample (for applicable modes)'}
        )
    self._mode = mode
    self._context_size = context_size
    assert(self._mode == 'rnn' or
           self._mode == 'ff',
           'TextSampler: invalid mode '..mode..' supplied.')
    if mode == 'rnn' then
        --nothing
    elseif mode == 'ff' then
        assert(self._context_size, 'ff sampling mode requires a context size')
    end
end

-- transform a batch to the appropriate form according to self._mode
-- text should already be set up to return an apropriate sequence length 
-- batches from text should be batch_size x sequence_length
function TextSampler:next_batch(text, batch_list)
    local batch = text:next_batch(batch_list)
    if self._mode == 'rnn' then
        local x = batch
        local y = torch.Tensor():typeAs(x):resizeAs(x)
        -- shifted copy
        y:sub(1,-1,1,-2):copy(x:sub(1,-1,2,-1))
        -- end of sentence
        y[{{},-1}] = text:end_id()
        return x, y
    elseif self._mode == 'ff' then
        local batch_seq_size = batch:size(2)
        local seq_size = self._context_size + 1
        assert(batch_seq_size >= seq_size, 'batch sequence length '..batch_seq_size..' too short for context of size '..seq_size)
        local trunc_cols = batch_seq_size - batch_seq_size % seq_size
        local reshaped = batch:narrow(2, 1, trunc_cols):reshape(batch:size(1)*trunc_cols/seq_size, seq_size)
        x = reshaped:sub(1,-1, 1,-2)
        y = reshaped:sub(1,-1, -1,-1)
        return x, y
    else
        error("What just happened? Unkown sampling mode in TextSampler.")
    end
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

