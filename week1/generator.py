# generator function which returns a meow the first time you call it, and then twice the number of meows on each consecutive call
def meow_generator():
    count = 1
    while True:
        yield "Meow " * count
        count *= 2

# testing generator
meow = meow_generator()
print(meow)
print(next(meow))
print(next(meow))
print(next(meow))
print(next(meow))
