from text_parser.text_parser_impl import TextParserImpl

text_parser = TextParserImpl()

text = "a yellow dog, a cute cat, a white lamp in front of the mirror, a red chair in front of the mirror"
text_2 = "a yellow dog, a red chair in front of the mirror"
print(text_parser.parse(text))
print(text_parser.parse(text_2))