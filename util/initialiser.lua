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
   --print(module.weight)
end

function M.initialise_biases(module)
   module.bias:zero()
   --print(module.bias)
end

function M.initialise_network(model, relu)
   -- hsm is different
   if not model.fg then hsm = true end
   model = model.fg and model or model.class_model
   model:apply(function(x)
      if x.weight then
         if hsm then print('WW',  x.weight:size()) end
         M.initialise_weights(x, relu)
      end
 end)
end

return M
