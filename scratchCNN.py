import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt

mndata = MNIST("Number_Samples")

xTrain, yTrain = mndata.load_training()
xTest, yTest = mndata.load_testing()

xTrain = np.array(xTrain)/255
xTest = np.array(xTest)/255
class ConvolutionalNeuralNetwork():
    
    def __init__(self, arch):
        self.kernels = np.random.uniform(-1,1,(16,3,3))
        self.weights = []
        self.biases = []
        self.size = len(arch) - 1
        for i in range(1, len(arch)):
            fan_in = arch[i-1]
            fan_out = arch[i]
            # Xavier initialization for weights
            limit = np.sqrt(7 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_out, fan_in)))
            self.biases.append(np.random.rand(arch[i], 1) - 0.5)

    def ReLU(self, x):
        return np.maximum(x, 0)

    def ReLUd(self, x):
        return x > 0

    def sigmoid(self, z):
        return 1/(1+np.exp(-(z)))

    def sigmoidd(self, x):
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))
    
    def oneHotter(self, x):
        new = np.zeros((10,1))
        new[x] = 1
        return new
    
    def loss(self, y_pred, y_true):
        epsilon = 1e-7  # small value to avoid division by zero
        loss = -np.sum(y_true.reshape(len(y_true)) * np.log(y_pred.reshape(len(y_pred)) + epsilon))
        return loss
    
    def forward(self, x):
        x = x.reshape(28,28)
        self.ps = []
        ksout = []
        self.backz = []
        for kernel in self.kernels: #for every kernel 
            new=[] #place to store the shrunkened result
            newps = []
            for yc in range(1,x.shape[1]-1): #every row that is to be kerneled
                row = [] #to store each row of it
                for xc in range(1,x.shape[0]-1): #the x position of that pixel that is going to be kerneled
                    #now we have the pixel selected, so let's get to kerneling
                    aaaa = 0
                    for k in range(-1,2): #y offset
                        for l in range(-1,2): #x offset
                            res = x[yc+k][xc+l]*kernel[k+1][l+1]
                            aaaa += res
                    aaaa = max(aaaa, 0)
                    row.append(aaaa)
                new.append(row)
            self.backz.append([[self.ReLUd(j) for j in i] for i in new])
            nn = []
            for i in range(0,len(new),2):
                nr = []
                nps =[]
                for j in range(0,len(new[0]), 2):
                    candids = [new[i][j]]
                    m1 = i + 1 < len(new)
                    m2 = j + 1 < len(new[0])
                    if m1 and m2:
                        candids += [new[i][j+1], new[i+1][j], new[i+1][j+1]]
                    elif m1:
                        candids += [0,new[i+1][j]]
                    elif m2:
                        candids.append(new[i][j+1])
                    nr.append(max(candids))
                    nps.append(np.argmax(candids))
                nn.append(nr)
                newps.append(nps)
            self.ps.append(newps)
            ksout.append(nn)
        newx = []
        for kernel in ksout:
            for row in kernel:
                newx += row
        self.zs = []
        self.aas = [np.array(newx).reshape(2704,1)]
        for i in range(self.size):
            self.zs.append(np.array([self.biases[i][jindex] + j for jindex, j in enumerate(self.weights[i].dot(self.aas[i]))]))#self.weights[i].dot(self.aas[i]) + self.biases[i])
            if i < self.size - 1:
                self.aas.append(self.sigmoid(self.zs[i]))
            else:
                self.aas.append(self.sigmoid(self.zs[i]))
    
    def givOut(self, x):
        self.forward(x.reshape(len(x),1))
        return self.aas[-1]
    
    def softmax(self, x):
        e = sum([np.exp(i) for i in x])
        return np.array([np.exp(i)/e for i in x])
    
    def backward(self, x, y):
        self.dws = [0]*self.size
        self.dbs = [0]*self.size
        for i in range(self.size-1,-1,-1):
            if i == self.size - 1: #Wt*dL/dY*dactive
                self.dbs[i] = 2 * (self.aas[i+1] - y) * self.sigmoidd(self.zs[i]) #dbs also stores dx
            else:
                self.dbs[i] = self.weights[i+1].T.dot(self.dbs[i+1])*self.sigmoidd(self.zs[i])
            self.dws[i] = self.dbs[i].dot(np.array(self.aas[i]).T) #dL/dY*Xt (dactivate already factored in)
        self.dLdP = [[[0]*26 for _ in range(26)] for _ in range(16)]
        dLdY = np.array(self.weights[0].T.dot(self.dbs[0])).reshape(16,13,13)
        for index, i in enumerate(dLdY):
            for rindex, row in enumerate(i):
                for eindex, elly in enumerate(row):
                    relpos = self.ps[index][rindex][eindex]
                    if relpos == 0:
                        self.dLdP[index][rindex*2][eindex*2] = elly*self.backz[index][rindex*2][eindex*2]
                    elif relpos == 1:
                        self.dLdP[index][rindex*2][eindex*2+1] = elly*self.backz[index][rindex*2][eindex*2+1]
                    elif relpos == 2:
                        self.dLdP[index][rindex*2+1][eindex*2] = elly*self.backz[index][rindex*2+1][eindex*2]
                    else:
                        self.dLdP[index][rindex*2+1][eindex*2+1] = elly*self.backz[index][rindex*2+1][eindex*2+1]
        self.dks = []
        for kernel in self.dLdP: #for every kernel 
            new=[] #place to store the shrunkened result
            for yc in range(0,3):
                row = []
                for xc in range(0,3):
                    aaaa = 0
                    for yy in range(0,26):
                        for xx in range(0,26):
                            aaaa += kernel[yy][xx]*x[yc+yy][xc+xx]
                    row.append(aaaa)
                new.append(row) 
            self.dks.append(new)
        


    def update(self, lr):
        for i in range(self.size):
            self.weights[i] -= self.dws[i] * lr
            self.biases[i] -= self.dbs[i] * lr
        for index, i in enumerate(self.kernels):
            for jindex, j in enumerate(i):
                for kindex, k in enumerate(j):
                    k -= self.dks[index][jindex][kindex]

    def descend(self, x, y, lr):
        self.forward(np.array(x).reshape(len(x), 1))
        self.backward(np.array(x).reshape(28, 28), self.oneHotter(y))
        self.update(lr)

    def train(self, x, y, lr, iter):
        print("Training begun")
        losses = []
        loss = 0
        correct = 0
        accuracies = []
        for t in range(iter):
            for i in range(len(x)):
                self.descend(x[i], y[i], lr)
                losss = self.loss(self.softmax(self.aas[-1]),self.oneHotter(y[i]))
                loss += losss
                print("Iteration number " + str(i+1))
                if np.argmax(self.aas[-1]) == y[i]:
                    correct += 1
                if i % 10 == 0:
                    losses.append(loss/10)
                    print("Avg loss is ", loss/10)
                    accuracies.append(correct/10)
                    loss = 0
                    correct = 0
            print(f"Epoch [{t+1}/{iter}]")
        print("Training finished")
        plt.plot(losses,color='red')
        plt.plot(accuracies,color='blue')
        plt.show()


    def test(self, x, y, iter):
        correct = 0
        loss = 0
        print("Testing begun")
        for i in range(iter):
            self.forward(np.array(x[i]).reshape(784, 1))
            #print("Iteration number " + str(i))
            lossa = self.loss(self.softmax(self.aas[-1]),self.oneHotter(y[i]))
            if np.argmax(self.aas[-1]) == y[i]:
                correct += 1
            print("For test #" + str(i))
            print(self.aas[-2])
            print(self.softmax(self.aas[-1]))
            print(self.oneHotter(y[i]))
            print(y[i])
            print(lossa)
            loss+=lossa
            print("------------------------------")
        print("Testing finished")
        print("The average loss in testing was " + str(loss/iter))
        return correct/iter 


nn = ConvolutionalNeuralNetwork([2704,64,32,10])
nn.train(xTrain, yTrain, 0.001, 2)
print(str(nn.test(xTest,yTest,100)*100)+"%")