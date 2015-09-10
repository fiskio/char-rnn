local SCRNN = {}

function SCRNN.scrnn(vocab_size, emb_size, hidden_size, context_size, n_layers, dropout, alpha)
   -- defaults
   emb_size = emb_size or 128
   hidden_size = hidden_size or 512
   context_size = context_size or 64
   n_layers = n_layers or 1
   dropout = dropout or 0
   alpha = alpha or 0.95
   -- there will be 2*n+1 inputs
   local inputs = {}
   table.insert(inputs, nn.Identity()()) -- x
   for L = 1, n_layers do
      table.insert(inputs, nn.Identity()()) -- prev_s[L]
      table.insert(inputs, nn.Identity()()) -- prev_h[L]
   end
   -- lookup table
   local dictionary = nn.LookupTable(vocab_size, emb_size)(inputs[1])
   -- build n layers
   local x, input_size_L
   local outputs = {}
   for L=1, n_layers do
      -- s,h from previos timesteps
      local prev_s = inputs[L*2]
      local prev_h = inputs[L*2+1]
      -- setup input to this layer
      if L == 1 then
         x = dictionary
         input_size_L = emb_size
      else
         x = outputs[(L-1)*2]
         if dropout > 0 then x = nn.Dropout(dropout)(x) end -- dropout?
         input_size_L = hidden_size
      end
      -- evaluate next context S
      local next_s = nn.CAddTable()({
         nn.MulConstant(1-alpha)(nn.Linear(input_size_L, context_size)(x)),
         nn.MulConstant(alpha)(prev_s)
      })
      -- evaluate next hidden H
      local next_h = nn.Sigmoid()({
         nn.CAddTable()({
            nn.Linear(input_size_L, hidden_size)(x),
            nn.Linear(context_size, hidden_size)(prev_s),
            nn.Linear(hidden_size, hidden_size)(prev_h)
         })
      })
      -- load outputs
      table.insert(outputs, next_s)
      table.insert(outputs, next_h)
   end
   -- fetch top S and H
   local top_s = outputs[#outputs-1]
   local top_h = outputs[#outputs]
   -- dropout?
   if dropout > 0 then
      top_s = nn.Dropout(dropout)(top_s)
      top_h = nn.Dropout(dropout)(top_h)
   end
   -- output layer
   local proj = nn.CAddTable()({
      nn.Linear(context_size, emb_size)(top_s),
      nn.Linear(hidden_size, emb_size)(top_h)
   })
   -- decoder
   local decoder = nn.Linear(emb_size, vocab_size)(proj)
   -- share embedding matrix
   -- dictionary.data.module:share(decoder.data.module, 'weight')
   -- softmax
   local logsoft = nn.LogSoftMax()(decoder)
   table.insert(outputs, logsoft)

   for k,v in pairs(dictionary) do print(k,v) end
   for k,v in pairs(decoder) do print(k,v) end
   return nn.gModule(inputs, outputs)
end

return SCRNN
