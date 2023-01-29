import math
import time


start = time.time()

a = []

num = 2

for i in range(num):
    a.append(i+1)


for element in a:
    print(a.index(element))

end = time.time()

print(end - start)