local CBOW = {}

--[[
function CBOW.create(vocab_size, context_size, output_class_size, hidden_size, emb_size, dropout, hsm)
   -- there are n+1 inputs (hiddens on each layer and x)
   local inputs = {nn.Identity()()}
   local encoder = nn.LookupTable(vocab_size, emb_size)(inputs[1])
   local sum = nn.Transpose()(nn.CAddTable()(nn.SplitTable(2)(nn.Transpose()(encoder))))
   local outputs = {sum}
   return nn.gModule(inputs, outputs)
end
--]]
function CBOW.create(vocab_size, context_size, output_class_size, hidden_size, emb_size, dropout, hsm)
-- there are n+1 inputs (hiddens on each layer and x)
   local inputs = {nn.Identity()()}
   local encoder = nn.LookupTable(vocab_size, emb_size)(inputs[1]):annotate{label='Embedding lookup'}
   local sum = nn.CAddTable()(nn.SplitTable(2)(encoder))
   local hidden = nn.ReLU()(nn.Linear(emb_size, emb_size)(sum))
   --local logsoft = LSM.lsm(output_class_size, hidden_size, emb_size, 0, hsm, encoder)(sum)
   local decoder = nn.Linear(emb_size, output_class_size)(hidden)
   --if dropout > 0 then decoder = nn.Dropout(dropout)(decoder) end
   local nnlogsoft = nn.LogSoftMax()(decoder)
   local outputs = {nnlogsoft}
   return nn.gModule(inputs, outputs)
end
return CBOW
