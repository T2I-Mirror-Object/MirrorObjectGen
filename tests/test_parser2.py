from text_parser.text_parser_impl_2 import TextParserImpl2

parser = TextParserImpl2()

# Test example 1
text1 = "A 3D rectangular table with a slanted top and base in front of a mirror, both the object and its perfect mirror reflection are visible"
result1 = parser.parse(text1)
print(f"Test 1:")
print(f"Input: {text1}")
print(f"Output: {result1}")
print(f"Expected: ['a 3D rectangular table with a slanted top and base']")
print(f"Pass: {result1 == ['a 3D rectangular table with a slanted top and base']}\n")

# Test example 2
text2 = "a dog, a cat, a chair in front of a mirror, both the objects and their reflections are visible"
result2 = parser.parse(text2)
print(f"Test 2:")
print(f"Input: {text2}")
print(f"Output: {result2}")
print(f"Expected: ['a dog', 'a cat', 'a chair']")
print(f"Pass: {result2 == ['a dog', 'a cat', 'a chair']}")
