require 'cudnn'
include 'Normalize.lua'
include 'StereoJoin.lua'

function save_network(dir, net)
   os.execute('mkdir -p ' .. dir)

   -- count conv layers
   local cnt = 0
   for i, m in ipairs(net.modules) do
      local t = torch.typename(m)
      if t == 'cudnn.SpatialConvolution' then
         cnt = cnt + 1
      end
   end

   local file = io.open(dir .. '/desc', 'w')
   file:write(('%d\n'):format(cnt))
   cnt = 0
   for i, m in ipairs(net.modules) do
      local t = torch.typename(m)
      if t == 'cudnn.SpatialConvolution' then
         local relu = i < net:size() and string.find(torch.typename(net.modules[i + 1]), 'ReLU') ~= nil
         file:write(('%d %d %d %d %d %d %d %d %d\n'):format(m.nInputPlane, m.nOutputPlane, m.kW, m.kH,
            m.dW, m.dH, m.padW, m.padH, relu and 1 or 0))
         torch.DiskFile(('%s/%dW'):format(dir, cnt), 'w'):binary():writeFloat(m.weight:float():storage())
         torch.DiskFile(('%s/%dB'):format(dir, cnt), 'w'):binary():writeFloat(m.bias:float():storage())
         cnt = cnt + 1
      end
   end
   file:close()
end

netname = arg[1]
net = torch.load(netname)[1]
dir = string.sub(netname, 1, -4)
print(net)
save_network(dir, net)
