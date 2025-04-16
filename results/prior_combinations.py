import itertools


# Define your arrays
array1 = [3, 0, -3, -4]
array2 = [10,15]
array3 = [-5,0,5]

# Generate all possible combinations
combinations = list(itertools.product(array1, array2, array3))

# Print the combinations
for combo in combinations:
    print(combo)

print(len(combinations))
