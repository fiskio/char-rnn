local CBOW = {}

function CBOW.create(vocab_size, context_size, output_class_size, hidden_size, emb_size, dropout, hsm)
   -- there are n+1 inputs (hiddens on each layer and x)
   local inputs = {nn.Identity()()}
   local encoder = nn.LookupTable(vocab_size, emb_size)(inputs[1]):annotate{label='Embedding lookup'}
   local sum = nn.CAddTable()(nn.SplitTable(1)(encoder))
   local logsoft = LSM.lsm(output_class_size, emb_size, emb_size, dropout, hsm, encoder)(sum)
   local outputs = {logsoft}
   return nn.gModule(inputs, outputs)
end

return CBOW
