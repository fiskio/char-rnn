local emoji = {}

emoji.all_emoji = {}
for line in io.lines('util/all_emoji.txt') do
   emoji.all_emoji[line] = true
end

function emoji.isEmoji(token)
   return emoji.all_emoji[token] and true or false
end

return emoji
