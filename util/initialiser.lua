-- Adapted from https://github.com/andyljones/char-rnn-experiments/blob/master/training.lua

local M = {}

function M.orthogonal_init(mat)
   assert(mat:size(1) == mat:size(2))
   local initScale = 1.1
   local M = torch.randn(mat:size(1), mat:size(1))
   local Q, R = torch.qr(M)
   local Q = Q:mul(initScale)
   mat:copy(Q)
end

function M.identity_init(mat)
   mat:eye(mat:size(1), mat:size(2))
end

function M.glorot_init(mat, relu)
   local relu = relu or false
   local n_in, n_out = mat:size(1), mat:size(2)
   local limit = math.sqrt(6/(n_in + n_out))
   if relu then
      limit = math.sqrt(2)*limit
   end
   mat:uniform(-limit, limit)
end

function M.initialise_weights(module, relu)
   local weights = module.weight
   if weights:size(1) == weights:size(2) then
      M.orthogonal_init(weights)
   else
      M.glorot_init(weights, relu)
   end
   print(module.weight)
end

function M.initialise_biases(module)
   module.bias:zero()
   print(module.bias)
end

function M.initialise_network(model, relu)
   --for k,v in pairs(model) do print(k) end
   local visited = {}
   for _,node in ipairs(model.fg.nodes) do
      local module = node.data.module
      if module and module.weight then
         M.initialise_weights(module, relu)
      end
      if module and module.bias then
         M.initialise_biases(module)
      end
   end
end

return M
