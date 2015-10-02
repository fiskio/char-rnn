local gpu_utils = {}

function gpu_utils.init()
   -- initialise cunn/cutorch for training on the GPU and fall back to CPU gracefully
   if opt.gpuid >= 0 and opt.opencl == 0 then
      local ok, cunn = pcall(require, 'cunn')
      local ok2, cutorch = pcall(require, 'cutorch')
      if not ok then print('Package cunn not found!') end
      if not ok2 then print('Package cutorch not found!') end
      if ok and ok2 then
         print('Using CUDA on GPU ' .. opt.gpuid .. '...')
         cutorch.setDevice(opt.gpuid+1)
         cutorch.manualSeed(opt.seed)
      else
         print('If cutorch and cunn are installed, your CUDA toolkit may be improperly configured.')
         print('Check your CUDA toolkit installation, rebuild cutorch and cunn, and try again.')
         print('Falling back on CPU mode')
         opt.gpuid = -1 -- overwrite user setting
      end
   end

   -- initialise clnn/cltorch for training on the GPU and fall back to CPU gracefully
   if opt.gpuid >= 0 and opt.opencl == 1 then
      local ok, cunn = pcall(require, 'clnn')
      local ok2, cutorch = pcall(require, 'cltorch')
      if not ok then print('Package clnn not found!') end
      if not ok2 then print('Package cltorch not found!') end
      if ok and ok2 then
         print('Using OpenCL on GPU ' .. opt.gpuid .. '...')
         cltorch.setDevice(opt.gpuid+1)
         torch.manualSeed(opt.seed)
      else
         print('If cltorch and clnn are installed, your OpenCL driver may be improperly configured.')
         print('Check your OpenCL driver installation, check output of clinfo command, and try again.')
         print('Falling back on CPU mode')
         opt.gpuid = -1 -- overwrite user setting
      end
   end
end

function gpu_utils.ship(tensor)
   if opt.gpuid >= 0 and opt.opencl == 0 then -- CUDA
      return tensor:cuda()
   end
   if opt.gpuid >= 0 and opt.opencl == 1 then -- OpenCL
      return tensor:cl()
   end
end 

function gpu_utils.ship_table(tbl)
   local out = {}
   for key, value in pairs(tbl) do
      out[key] = gpu_utils.ship(value)
   end
   return out
end

return gpu_utils
