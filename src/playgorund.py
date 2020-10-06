import nltk

command="robot, please pick up the box"
text = nltk.word_tokenize(command)
pos_tags = nltk.pos_tag(text)
print(pos_tags)