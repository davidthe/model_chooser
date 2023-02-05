from Levenshtein import ratio, distance

# Calculates a normalized indel similarity in the range [0, 1]. This is calculated as 1 - normalized_distance


print(ratio("אני שמן מא", "אני שמן מאוד"))
print(distance("אני שמן", "אני שמן מאוד"))

filename = './16-vat44-gt.txt'

with open(filename) as file:
    while (line := file.readline().rstrip()):
        print(line)