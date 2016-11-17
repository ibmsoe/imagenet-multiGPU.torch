--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
local Threads = require 'threads'
Threads.serialization('threads.sharedserialize')


--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,    LR,   WD,
        {  1,     18,   1e-2,   5e-4, },
        { 19,     29,   5e-3,   5e-4  },
        { 30,     43,   1e-3,   0 },
        { 44,     52,   5e-4,   0 },
        { 53,    1e8,   1e-4,   0 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local top1_epoch, loss_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk

local batchSize_l = opt.batchSize
local epochSize_l = opt.epochSize
local nClasses = opt.nClasses
logger_thread = Threads(1,function() require 'cutorch' end, 
                          function()
        local batchSize_l = batchSize_l
        local epochSize_l = epochSize_l
 end)
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = 0.0,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   top1_epoch = 0
   loss_epoch = 0
   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, labels = trainLoader:sample(opt.batchSize)
            return inputs, labels
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()
   logger_thread:synchronize()

   top1_epoch = top1_epoch * 100 / (opt.batchSize * opt.epochSize)
   loss_epoch = loss_epoch / opt.epochSize

   trainLogger:add{
      ['% top1 accuracy (train set)'] = top1_epoch,
      ['avg loss (train set)'] = loss_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average loss (per batch): %.2f \t '
                          .. 'accuracy(%%):\t top-1 %.2f\t',
                       epoch, tm:time().real, loss_epoch, top1_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()
local labels = torch.CudaTensor()

local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, labelsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   labels:resize(labelsCPU:size()):copy(labelsCPU)

   local err, outputs
   feval = function(x)
      model:zeroGradParameters()
      local outputs = model:forward(inputs)
      logger_thread:synchronize()
      logger_thread:addjob(function() output_l = outputs end)
      err = criterion:forward(outputs, labels)
      local gradOutputs = criterion:backward(outputs, labels)
      model:backward(inputs, gradOutputs)
      return err, gradParameters
   end
   optim.sgd(feval, parameters, optimState)

   cutorch.synchronize()
   batchNumber = batchNumber + 1
   loss_epoch = loss_epoch + err
   
  local function sortLog(time_l,epoch_l,batchNumber_l,dataLoadingTime_l,err_l,lr_l)
     local top1 = 0
     local top1_r
     do
        local _,prediction_sorted = output_l:float():sort(2, true) -- descending
        for i=1,batchSize_l do
          inRangeLabel = ((prediction_sorted[i][1]-1) % nClasses) + 1
          if inRangeLabel == labelsCPU[i] then
           top1 = top1 + 1
          end
        end
        top1_r = top1 * 100 / batchSize_l;
     end
     print(('Epoch: [%d][%d/%d]\tTime %.3f Err %.4f Top1-%%: %.2f LR %.0e DataLoadingTime %.3f'):format(
            epoch_l, batchNumber_l, epochSize_l, time_l, err_l, top1_r, lr_l, dataLoadingTime_l))

     return top1
  end

  local time_l = timer:time().real
  local epoch_l = epoch
  local batchNumber_l = batchNumber
  local dataLoadingTime_l = dataLoadingTime;
  local err_l = err
  local lr_l = optimState.learningRate
  logger_thread:addjob(function() return sortLog(time_l,epoch_l,batchNumber_l,dataLoadingTime_l,err_l,lr_l) end, 
                       function(t) top1_epoch = top1_epoch + t end)
   
  dataTimer:reset()
end
