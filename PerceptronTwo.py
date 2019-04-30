import numpy as np
import matplotlib.pyplot as plt
from random import randint

#TODO: there a a couple of places for optimization:
#1)Calculate accuracy only once per epoch
#2)Incorporate testing accuracy within learning

#Entry point into the program
class MLP:

	def __init__(self, hiddenNeurons, beta, momentum, learning, epochs, batchSize):
		#constructor
		self.hiddenNeurons = hiddenNeurons
		self.beta = beta
		self.momentum = momentum
		self.learning = learning
		self.epochs = epochs
		self.batchSize = batchSize

	#Helper function that loads and filters train/test data and
	#creates weights (basically initializes data we need)
	def getData(self):
		#open files with train/test data -> store into matrices
		rawTrain = np.loadtxt(open("mnist_train.csv", "rb"), delimiter=",", skiprows=1)
		rawTest = np.loadtxt(open("mnist_test.csv", "rb"), delimiter=",", skiprows=1)

		#grab the target columns and create somewhat of a
		#one hot vector (matrix) with 0.9 and 0.1 instead of 1 and 0
		trainTargets = rawTrain[:,0:1] 
		testTargets = rawTest[:,0:1] 

		self.trainTargets = np.full((rawTrain.shape[0], 10), 0.1)
		self.testTargets = np.full((rawTest.shape[0], 10), 0.1)

		rowNumber = 0
		for target in trainTargets:
			self.trainTargets[rowNumber, int(target)] = 0.9
			rowNumber +=1

		rowNumber = 0
		for target in testTargets:
			self.testTargets[rowNumber, int(target)] = 0.9
			rowNumber+=1

		#remove first column from the input datas (target column)
		rawTrain = np.delete(rawTrain, 0, 1)
		rawTest = np.delete(rawTest, 0, 1)

		#optimize data 
		rawTrain = rawTrain / 255
		rawTest = rawTest / 255

		#create bias vector and concat bias into data
		trainRows = rawTrain.shape[0]
		testRows = rawTest.shape[0]

		trainBias = np.full((1, trainRows), 1)
		testBias = np.full((1, testRows), 1)

		rawTrain = np.concatenate((rawTrain, trainBias.T), axis = 1)
		rawTest = np.concatenate((rawTest, testBias.T), axis = 1)

		#create actual train/test matrices
		self.trainData = rawTrain
		self.testData = rawTest

		#make weights
		cols = self.trainData.shape[1]

		self.weights1 = np.random.uniform(low=-0.5, high = 0.5, size=(cols, self.hiddenNeurons))  #input -> hidden
		self.weights2 = np.random.uniform(low=-0.5, high = 0.5, size=(self.hiddenNeurons+1, 10))  #hidden -> output

	#Function that performs the training by grabbing random batches from the dataset
	def trainRandom(self):
		updatew1 = np.zeros((np.shape(self.weights1)))
		updatew2 = np.zeros((np.shape(self.weights2)))

		self.weightList1 = []  #list that stores weights1 (1 per epoch)
		self.weightList2 = []  #list that stores weights2 (1 per epoch)
		self.trainAccuracyList = []  #list that stores accuracy during the training (1 accuracy per epoch)

		for epoch in range(self.epochs):
			for r in range(self.batchSize):
				batchStart = self.getRandStart()  #grabs random number b/w 0-60k -> where batch will start
				batchEnd = batchStart + self.batchSize  #where batch will end (depending on the batch size)

				#batch matrix used for this epoch
				inputMatrix = self.trainData[batchStart:batchEnd, :]  #e.g. 20 x 785 or batchSize x 785

				self.outputs = self.forward(inputMatrix)  #forward step (calls a helper function)

				batchTargets = self.trainTargets[batchStart:batchEnd, :]  #grab targets for this particular batch
				correct = 0;
				#Calculate accuracy for this batch
				for i in range (self.batchSize):
					out = np.argmax(self.outputs[i:i+1, :])
					exp = np.argmax(batchTargets[i:i+1, :])

					if out == exp: correct+=1

				accuracy = (correct / self.batchSize) * 100
				error = 0.5*np.sum((self.outputs-self.trainTargets[batchStart:batchEnd, :])**2)
				print ("Training: ",epoch, "Accuracy:", accuracy, "%", "Error: ",error) 

				#BACKWARD propogation (step):

				#deltao = self.beta*(self.outputs-batchTargets)*self.outputs*(1.0-self.outputs)
				deltao = (self.outputs-batchTargets)*(self.outputs*(-self.outputs)+self.outputs)/self.batchSize

				deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
				          
				updatew1 = self.learning*(np.dot(np.transpose(inputMatrix),deltah[:,:-1])) + self.momentum*updatew1
				updatew2 = self.learning*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2

				weights1 = self.weights1 - updatew1
				weights2 = self.weights2 - updatew2

				self.weights1 = weights1
				self.weights2 = weights2

			self.trainAccuracyList.append(accuracy)  #insert latest accuracy
			self.weightList1.append(weights1)  #insert latest weight1
			self.weightList2.append(weights2)  #insert latest weight2

		#Once we are done with training, call functions for testing and show output
		self.test()
		self.accuracyGraph()
		self.confMat()

	#Function that uses incremental batches (not fully tested)
	def trainWhole(self):
		updatew1 = np.zeros((np.shape(self.weights1)))
		updatew2 = np.zeros((np.shape(self.weights2)))

		self.weightList1 = []
		self.weightList2 = []
		self.trainAccuracyList = []

		increase = int(self.trainData.shape[0] / self.batchSize)  #e.g. 20 batches = 3000 rows per batch
		print("BatchSize:", increase)

		for epoch in range(self.epochs):
			for i in range(self.batchSize):
				batchStart = i * increase
				batchEnd = batchStart + increase 

				print("Start:", batchStart)
				print("End", batchEnd)

				#batch matrix used for this epoch
				inputMatrix = self.trainData[batchStart:batchEnd, :]  #e.g. 20 x 785 or batchSize x 785

				self.outputs = self.forward(inputMatrix)

				batchTargets = self.trainTargets[batchStart:batchEnd, :]
				correct = 0;
				for j in range (increase):
					out = np.argmax(self.outputs[j:j+1, :])
					exp = np.argmax(batchTargets[j:j+1, :])

					if out == exp: correct+=1

				accuracy = (correct / increase) * 100
				error = 0.5*np.sum((self.outputs-self.trainTargets[batchStart:batchEnd, :])**2)
				print ("Training: ",epoch, "Accuracy:", accuracy, "%", "Error: ",error) 

				#deltao = self.beta*(self.outputs-batchTargets)*self.outputs*(1.0-self.outputs)
				deltao = (self.outputs-batchTargets)*(self.outputs*(-self.outputs)+self.outputs)/self.batchSize

				deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
				          
				updatew1 = self.learning*(np.dot(np.transpose(inputMatrix),deltah[:,:-1])) + self.momentum*updatew1
				updatew2 = self.learning*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2

				weights1 = self.weights1 - updatew1
				weights2 = self.weights2 - updatew2

				self.weights1 = weights1
				self.weights2 = weights2

			self.trainAccuracyList.append(accuracy)
			self.weightList1.append(weights1)
			self.weightList2.append(weights2)

		self.test()
		self.accuracyGraph()
		self.confMat()

	#Function that tests the network
	#We are using different weights from each epoch for output purposes (show how the net learned)
	def test(self):
		self.testAccuracyList = []  #list that holds accuracy of test per epoch

		for epoch in range (0, self.epochs):
			
			correct = 0

			#forward step
			in_to_hidden = np.dot(self.testData, self.weightList1[epoch])
			in_to_hidden = 1.0 / (1.0 + np.exp(-self.beta * in_to_hidden))

			#concat bias to hidden
			hiddenBias = np.full((1, in_to_hidden.shape[0]), 1)
			in_to_hidden = np.concatenate((in_to_hidden, hiddenBias.T), axis = 1)

			hidden_to_out = np.dot(in_to_hidden, self.weightList2[epoch])
			hidden_to_out = 1.0 / (1.0 + np.exp(-self.beta * hidden_to_out))

			#compute accuracy for this particular set of weights
			for i in range (0, self.testData.shape[0]):
				out = np.argmax(hidden_to_out[i:i+1, :])
				exp = np.argmax(self.testTargets[i:i+1, :])

				if out == exp: 
					correct += 1

			accuracy = (correct / self.testData.shape[0]) * 100
			print("Test:", epoch, "Accuracy:", accuracy, "%")
			self.testAccuracyList.append(accuracy)  #insert accuracy into the list

	#helper function to get random number for a batch
	def getRandStart(self):
		start = randint(0, self.trainData.shape[0])

		while start >= (self.trainData.shape[0] - self.batchSize + 1):
			start = randint(0, self.trainData.shape[0])

		return start

	#Helper function for forward step
	def forward(self, inputMatrix):
		#calculate hidden layer input
		self.hidden = np.dot(inputMatrix, self.weights1)  #compute dot product for lvl1
		self.hidden = 1.0 / (1.0 + np.exp(-self.beta * self.hidden))

		#concat bias to hidden
		hiddenBias = np.full((1, self.hidden.shape[0]), 1)
		self.hidden = np.concatenate((self.hidden, hiddenBias.T), axis = 1)

		#calculate output
		output = np.dot(self.hidden, self.weights2)
		output = 1.0 / (1.0 + np.exp(-self.beta * output))

		return output

	
	#helper function that graphs accuracy per epoch for both training
	#and testing data
	def accuracyGraph(self):
		plt.title("Accuracy Graph")
		plt.xlabel("Epoch")
		plt.ylabel("Accuracy")
		plt.axis([0,self.epochs,0,100])

		epochList = []
		for i in range (0, self.epochs):
			epochList.append(i)

		plt.plot(epochList, self.trainAccuracyList, color = "green", label="Training")
		plt.plot(epochList, self.testAccuracyList, color = "red", label="Test")
		plt.legend()
		plt.show()

	#helper function that computes the confusion matrix for the testing
	#data using the latest (most accurate) weights
	def confMat(self):
		#create conf matrix
		size = self.testTargets.shape[1]
		conf = np.zeros(shape=(size+1, size+1), dtype = int)
		corr = 0

		for i in range(size):
			conf[0,i+1] = i
			conf[i+1,0] = i

		in_to_hidden = np.dot(self.testData, self.weights1)
		in_to_hidden = 1.0 / (1.0 + np.exp(-self.beta * in_to_hidden))

		#concat bias to hidden
		hiddenBias = np.full((1, in_to_hidden.shape[0]), 1)
		in_to_hidden = np.concatenate((in_to_hidden, hiddenBias.T), axis = 1)

		hidden_to_out = np.dot(in_to_hidden, self.weights2)
		hidden_to_out = 1.0 / (1.0 + np.exp(-self.beta * hidden_to_out))

		correct = 0

		for i in range(0, self.testData.shape[0]):

			out = np.argmax(hidden_to_out[i:i+1, :])
			exp = np.argmax(self.testTargets[i:i+1, :])

			if out == exp: 
				correct += 1

			conf[exp+1, out+1] += 1

		print(conf)
		print("Accuracy:", (correct / self.testData.shape[0]) * 100, "%")
		print("Learning rate:", self.learning)
		print("Hidden neurons:", self.hiddenNeurons)
		print("Momentum:", self.momentum)
		print("Epochs:", self.epochs)
		print("Batch size:", self.batchSize)

t2 = MLP(100, 1, 0.9, 0.1 , 50, 100)
t2.getData()
t2.trainRandom() 

