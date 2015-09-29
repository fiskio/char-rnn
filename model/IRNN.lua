local IRNN = {}

function IRNN.rnn(input_size, rnn_size, n_layers, emb_size, dropout, hsm, sharing)
   -- there are n+1 inputs (hiddens on each layer and x)
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   for L = 1,n_layers do
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end
   -- setup
   local outputs = {}
   local x, input_size_L
   local encoder = nn.LookupTable(input_size, emb_size)(inputs[1])
   -- for all layers
   for L = 1,n_layers do
      local prev_h = inputs[L+1]
      if L == 1 then
         x = encoder
         input_size_L = emb_size
      else
         x = outputs[(L-1)]
         if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
         input_size_L = rnn_size
      end
      -- IRNN input transforms
      local i2h = nn.Linear(input_size_L, rnn_size)(x)
      local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
      -- IRNN initialisation
      h2h.data.module.weight = torch.eye(rnn_size)
      h2h.data.module.bias = torch.zeros(rnn_size)
      -- IRNN non linearity
      local next_h = nn.ReLU()(nn.CAddTable(){i2h, h2h})
      table.insert(outputs, next_h)
   end
   -- dropout?
   local top_h = outputs[#outputs]
   if dropout > 0 then
      top_h = nn.Dropout(dropout)(top_h)
   else
      top_h = nn.Identity()(top_h)
   end
   -- HSM?
   if hsm == 0 then
      -- no hsm
      local proj = nn.Linear(rnn_size, emb_size)(top_h)
      local decoder = nn.Linear(emb_size, input_size)(proj)
      -- sharing encoder/decoder?
      if sharing then
         encoder.data.module:share(decoder.data.module, 'weight', 'gradWeight')
      end
      local logsoft = nn.LogSoftMax()(decoder)
      table.insert(outputs, logsoft)
   else
      -- hsm
      table.insert(outputs, top_h)
   end

   return nn.gModule(inputs, outputs)
end

return IRNN
