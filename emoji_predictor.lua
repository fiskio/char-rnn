require 'torch'
require 'nn'
require 'nngraph'
require 'optim'
require 'lfs'

require 'util.OneHot'
require 'util.misc'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Sample from a character-level language model')
cmd:text()
cmd:text('Options')
-- required:
cmd:argument('-model','model checkpoint to use for sampling')
-- optional parameters
cmd:option('-seed',123,'random number generator\'s seed')
cmd:option('-npreds',5,'number of emoji predictions')
cmd:option('-gpuid',0,'which gpu to use. -1 = use CPU')
cmd:option('-opencl',0,'use OpenCL (instead of CUDA)')
cmd:option('-verbose',0,'set to 0 to ONLY print the sampled text, no diagnostics')
cmd:option('-emojis','util/emoji_all.txt','emoji list')
cmd:option('-interface', 'console', 'console | web')
cmd:text()

-- parse input params
opt = cmd:parse(arg)

-- gated print: simple utility function wrapping a print
function gprint(str)
    if opt.verbose == 1 then print(str) end
end

-- check that cunn/cutorch are installed if user wants to use the GPU
if opt.gpuid >= 0 and opt.opencl == 0 then
    local ok, cunn = pcall(require, 'cunn')
    local ok2, cutorch = pcall(require, 'cutorch')
    if not ok then gprint('package cunn not found!') end
    if not ok2 then gprint('package cutorch not found!') end
    if ok and ok2 then
        gprint('using CUDA on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cutorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        cutorch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

-- check that clnn/cltorch are installed if user wants to use OpenCL
if opt.gpuid >= 0 and opt.opencl == 1 then
    local ok, cunn = pcall(require, 'clnn')
    local ok2, cutorch = pcall(require, 'cltorch')
    if not ok then print('package clnn not found!') end
    if not ok2 then print('package cltorch not found!') end
    if ok and ok2 then
        gprint('using OpenCL on GPU ' .. opt.gpuid .. '...')
        gprint('Make sure that your saved checkpoint was also trained with GPU. If it was trained with CPU use -gpuid -1 for sampling as well')
        cltorch.setDevice(opt.gpuid + 1) -- note +1 to make it 0 indexed! sigh lua
        torch.manualSeed(opt.seed)
    else
        gprint('Falling back on CPU mode')
        opt.gpuid = -1 -- overwrite user setting
    end
end

torch.manualSeed(opt.seed)

-- load the model checkpoint
if not lfs.attributes(opt.model, 'mode') then
    gprint('Error: File ' .. opt.model .. ' does not exist. Are you sure you didn\'t forget to prepend cv/ ?')
end
checkpoint = torch.load(opt.model)
protos = checkpoint.protos
protos.rnn:evaluate() -- put in eval mode so that dropout works properly

-- initialize the vocabulary (and its inverted version)
local vocab = checkpoint.vocab
local ivocab = {}
for c,i in pairs(vocab) do ivocab[i] = c end

-- load emojis
local emoji_list = {}
for e in io.lines(opt.emojis) do
   emoji_list[e] = true
end

-- PREDICTOR
-- returns the most likely emojis for a given text
function predict_emoji(seed_text, npreds)

   -- initialize the rnn state to all zeros
   gprint('creating an ' .. checkpoint.opt.model .. '...')
   local current_state
   current_state = {}
   for L = 1,checkpoint.opt.num_layers do
      -- c and h for all layers
      local h_init = torch.zeros(1, checkpoint.opt.rnn_size):double()
      if opt.gpuid >= 0 and opt.opencl == 0 then h_init = h_init:cuda() end
      if opt.gpuid >= 0 and opt.opencl == 1 then h_init = h_init:cl() end
      table.insert(current_state, h_init:clone())
      if checkpoint.opt.model == 'lstm' then
         table.insert(current_state, h_init:clone())
      end
   end
   state_size = #current_state

   -- do a few seeded timesteps
   local prediction
   if string.len(seed_text) > 0 then
      gprint('seeding with ' .. seed_text)
      gprint('--------------------------')
      for c in seed_text:gmatch'.' do
         prev_char = torch.Tensor{vocab[c]}
         if opt.gpuid >= 0 and opt.opencl == 0 then prev_char = prev_char:cuda() end
         if opt.gpuid >= 0 and opt.opencl == 1 then prev_char = prev_char:cl() end
         local lst = protos.rnn:forward{prev_char, unpack(current_state)}
         -- lst is a list of [state1,state2,..stateN,output]. We want everything but last piece
         current_state = {}
         for i=1,state_size do table.insert(current_state, lst[i]) end
         prediction = lst[#lst] -- last element holds the log probabilities
      end
   else
      -- fill with uniform probabilities over characters (? hmm)
      gprint('missing seed text, using uniform probability over first character')
      gprint('--------------------------')
      prediction = torch.Tensor(1, #ivocab):fill(1)/(#ivocab)
      if opt.gpuid >= 0 and opt.opencl == 0 then prediction = prediction:cuda() end
      if opt.gpuid >= 0 and opt.opencl == 1 then prediction = prediction:cl() end
   end

   -- fetch top emoji predictions
   local probs, indices = torch.sort(prediction, true)
   local emoji_preds = {}
   for i=1,prediction:size(2) do
      if #emoji_preds >= npreds then break end
      local char = ivocab[indices[1][i]]
      if emoji_list[char] then
         local pair = { [char] = i }
         table.insert(emoji_preds, pair)
      end
   end
   return emoji_preds
end

if opt.interface == 'console' then
   while(true) do
      local line = io.read("*line")
      if not line then break end
      local p = predict_emoji(line, opt.npreds)
      print(p)
   end
elseif opt.interface == 'web' then

   local turbo = require("turbo")

   local EmojiPredictionHandler = class("EmojiPredictionHandler", turbo.web.RequestHandler)

   function EmojiPredictionHandler:post()
      -- Get the 'name' argument, or use 'Easter Bunny' if it does not exist
      local name = self:get_argument("text", "")
      local npreds = self:get_argument("npreds", opt.npreds)
      local p = predict_emoji('ciao', tonumber(npreds))
      self:write(p)
   end

   local app = turbo.web.Application:new({
      {"/emojis", EmojiPredictionHandler}
   })

   app:listen(8888)
   turbo.ioloop.instance():start()
else
   print('-interface can be console or web')
end
