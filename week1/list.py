# get a list of the squares of each number between 0 and 100
squares = [x**2 for x in range(101)]
print(squares)
# get a list of the squares of each even number between 0 and 100
squares_even = [x**2 for x in range(101) if x % 2 == 0]
print(squares_even)