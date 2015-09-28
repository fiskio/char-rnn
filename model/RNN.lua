local RNN = {}

function RNN.rnn(input_size, rnn_size, n_layers, embeddings, dropout, hsm)

  -- there are n+1 inputs (hiddens on each layer and x)
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x
  for L = 1,n_layers do
    table.insert(inputs, nn.Identity()()) -- prev_h[L]

  end

  local x, input_size_L
  local outputs = {}
  for L = 1,n_layers do

    local prev_h = inputs[L+1]
    if L == 1 then
      x = nn.LookupTable(input_size, embeddings)(inputs[1])
      input_size_L = embeddings
    else
      x = outputs[(L-1)]
      if dropout > 0 then x = nn.Dropout(dropout)(x) end -- apply dropout, if any
      input_size_L = rnn_size
    end

    -- RNN tick
    local i2h = nn.Linear(input_size_L, rnn_size)(x)
    local h2h = nn.Linear(rnn_size, rnn_size)(prev_h)
  --  if i2h.data.module.weight:size() ~= h2h.data.module.weight:size() then
       print(i2h.data.module.weight:size(), h2h.data.module.weight:size())
    --end
    local next_h = nn.Tanh()(nn.CAddTable(){i2h, h2h})

    table.insert(outputs, next_h)
  end
   -- set up the decoder
   local top_h = outputs[#outputs]
   if dropout > 0 then top_h = nn.Dropout(dropout)(top_h) end
   local proj = nn.Linear(rnn_size, embeddings)(top_h)
   -- HSM?
   if hsm == 0 then
      local final = nn.Linear(embeddings, input_size)(proj)
      local logsoft = nn.LogSoftMax()(final)
      table.insert(outputs, logsoft)
   else
      table.insert(outputs, proj)
   end

  return nn.gModule(inputs, outputs)
end

return RNN
