local stx = require 'pl.stringx'
local emoji = {}

emoji.all_emoji = {}
for line in io.lines('all_emoji.txt') do
   emoji.all_emoji[stx.strip(line)] = true
end
print(emoji)
function emoji.isEmoji(token)
   return emoji.all_emoji[token] and true or false
end

return emoji
