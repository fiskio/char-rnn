local LSM = {}

function LSM.lsm(input_size, rnn_size, emb_size, dropout, hsm, encoder)
   -- there are n+1 inputs (hiddens on each layer and x)
   local inputs = {}
   table.insert(inputs, nn.Identity()())
   -- setup
   local outputs = {}
   -- dropout?
   local top_h = inputs[1]
   if dropout > 0 then
      top_h = nn.Dropout(dropout)(top_h)
   else
      top_h = nn.Identity()(top_h)
   end
   local proj = nn.Linear(rnn_size, emb_size)(top_h)
   -- HSM?
   if hsm == 0 then
      -- no hsm
      decoder = nn.Linear(emb_size, input_size)(proj)
      local logsoft = nn.LogSoftMax()(decoder)
      table.insert(outputs, logsoft)
   else
      -- hsm
      table.insert(outputs, proj)
   end

   local out = nn.gModule(inputs, outputs)
   out.name = 'lsm'
   out.encoder = encoder
   out.decoder = decoder
   return out
end

return LSM
