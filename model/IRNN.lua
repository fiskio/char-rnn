local IRNN = {}

function IRNN.rnn(input_size, rnn_size, n_layers, emb_size, dropout, hsm)
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
   -- softmax
   local top_h = outputs[#outputs]
   local logsoft = LSM.lsm(input_size, rnn_size, emb_size, dropout, hsm, encoder)(top_h)
   table.insert(outputs, logsoft)

   return nn.gModule(inputs, outputs)
end

return IRNN
