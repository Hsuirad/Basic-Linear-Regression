import numpy
from random import random
from random import seed
import matplotlib.pyplot as plt

LEARNING_RATE = 0.01
TRAINING_RUNS = 2000


'''
-----------------------------
        MAIN PROBLEM

      ALPHA IS TOO HIGH
-----------------------------
'''

class Regression:

	def __init__(self, points):
		self.points = points	
		self.alpha = 0.0001
		self.inputs = [1, 0] #x's
		self.weights = [0, 0] #thetas

	def train(self):
		for j in range(TRAINING_RUNS):
			for i in self.points:
				x_i, y_i = i #input is a (x, y) index of self.points

				self.inputs[1] = x_i

				h_1 = h_2 = h_3 = 0
			
				h_1 = 1 * self.weights[0]
				h_2 = x_i * self.weights[1]
				h_3 = h_1 + h_2
					
				self.weights[0] = self.weights[0] + self.alpha * (y_i - h_3) * 1

				h_1 = 1 * self.weights[0]
				h_2 = x_i * self.weights[1]
				h_3 = h_1 + h_2

				self.weights[1] = self.weights[1] + self.alpha * (y_i - h_3) * self.inputs[1]


	def return_line(self):
		return self.weights



points = []

for i in range(25):
    a = i 
    b = i + (random()*5-5) + 10

    points.append((a, b))

    print(points[i])

test = Regression(points)

test.train()

line = test.return_line()

print(line)

x = []
y = []
for i in points:
    z, w = i
    x.append(z)
    y.append(w)

plt.scatter(x, y)

plt.plot(numpy.linspace(0,numpy.argmax(x), 100), numpy.linspace(0,numpy.argmax(x), 100)*line[1] + line[0])

print(points)
plt.show()
