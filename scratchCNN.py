import numpy as np
from mnist import MNIST
import matplotlib.pyplot as plt
import math

np.random.seed(1337)

mndata = MNIST("Number_Samples")

xTrain, yTrain = mndata.load_training()
xTest, yTest = mndata.load_testing()

xTrain = np.array(xTrain)/255
xTest = np.array(xTest)/255
class ConvolutionalNeuralNetwork():
    
    def __init__(self, insize, convarch, arch):
        self.kernels = [] #list of kernels, [numberkernels, crosssize, poolsize]
        self.convarch = convarch
        numouts = 1
        outsize = insize
        self.outsizes = [outsize]
        self.poutsizes = []
        for i in convarch:
            self.kernels.append(np.random.uniform(-1,1,(i[0],i[1],i[1])))
            numouts *= i[0]
            poutsize = [outsize[0]-i[1]+1, outsize[1]-i[1]+1]
            outsize = [math.ceil(j/i[2]) for j in poutsize]
            self.outsizes.append(outsize)
            self.poutsizes.append(poutsize)
        arch.insert(0, numouts*outsize[0]*outsize[1])
        self.weights = []
        self.biases = []
        self.outputSize = arch[-1]
        self.size = len(arch) - 1
        for i in range(1, len(arch)):
            fan_in = arch[i-1]
            fan_out = arch[i]
            #Xavier for weights
            limit = np.sqrt(7 / (fan_in + fan_out))
            self.weights.append(np.random.uniform(-limit, limit, (fan_out, fan_in)))
            self.biases.append(np.random.rand(arch[i], 1) - 0.5)

    def ReLU(self, x):
        return np.maximum(x, 0)

    def ReLUd(self, x): #derivative of ReLU
        return x > 0

    def sigmoid(self, z):
        return 1/(1+np.exp(-(z)))

    def sigmoidd(self, x): #derivative of sigmoid
        return self.sigmoid(x) * (1.0 - self.sigmoid(x))
    
    def oneHotter(self, x): #one hot encoder for output
        new = np.zeros((self.outputSize,1))
        new[x] = 1
        return new
    
    def loss(self, y_pred, y_true):
        epsilon = 1e-7  #just so no division by zero
        loss = -np.sum(y_true.reshape(len(y_true)) * np.log(y_pred.reshape(len(y_pred)) + epsilon))
        return loss
    
    def validCrossCorr(self, x, f, pp=False):
        if pp:
            print(x)
            print(f)
        res = np.zeros((len(x)-len(f)+1, len(x[0])-len(f[0])+1))
        for i in range(len(x)-len(f)+1):
            for j in range(len(x[0])-len(f[0])+1):
                res[i][j] = np.sum(np.multiply(f, x[i:i+len(f),j:j+len(f[0])]))
        backz = np.copy(res)
        res = [[self.ReLU(j) for j in i] for i in res]
        backz = [[self.ReLUd(j) for j in i] for i in backz]
        return res, backz
    
    def fullConvolve(self, d, f):
        newf = np.flip(f, (0,1))
        padding_width = ((len(f)-1,len(f)-1),(len(f[0])-1,len(f[0])-1))
        newd = np.pad(d, pad_width=padding_width, mode="constant", constant_values=0)
        res, _ = self.validCrossCorr(newd, newf)
        return res
    
    def maxpool(self, x, s):
        res = np.zeros((math.ceil(len(x)/s), math.ceil(len(x[0])/2)))
        pos = np.copy(res)
        for i in range(0,len(x),s):
            for j in range(0,len(x[0]),s):
                cans = x[i:min(len(x),i+s), j:min(len(x[0]), j+s)]
                cans = np.array([np.append(k, [0]*(s-len(k))) for k in cans]).reshape(-1)
                res[int(i/s),int(j/s)] = max(cans)
                pos[int(i/s),int(j/s)] = np.argmax(cans)
        return res, pos

    def unpool(self, x, s, pos, osx, osy):
        res = np.zeros((osy,osx))
        for i in range(len(x)):
            for j in range(len(x[0])):
                ri = (i*s)+(pos[i,j]//s)
                rj = (j*s)+(pos[i,j]%s)
                res[int(ri), int(rj)] = x[i,j]
        return res
     
    def softmax(self, x):
        e = sum([np.exp(i) for i in x])
        return np.array([np.exp(i)/e for i in x])
    
    def forward(self, x):
        self.ps = []
        self.ksout = [[[x]]] # (layer, kernel, input, row, value)
        self.backz = [] #store for backprop
        for ln, layer in enumerate(self.kernels):
            zl = [] #zstore for layer
            pl = [] #posstore for layer
            newks = []
            for kernel in layer: #for every kernel 
                zinp = []
                pinp = []
                ksi = []
                for inp in self.ksout[-1]: #for every input from the previous layer
                    p, z = self.validCrossCorr(inp, kernel) #product and zstore of the crosscorrelation
                    zinp.append(z) #intermediate zstore
                    y, ps = self.maxpool(np.array(p), self.convarch[ln][2]) #maxpool the output, store positions
                    ksi.append(y) #ksout add the output
                    pinp.append(ps) #store the positions
                zl.append(zinp)
                pl.append(pinp)
                newks.append(ksi)
            self.ksout.append(newks)
            self.backz.append(zl)
            self.ps.append(pl)
        self.zs = []
        self.aas = [np.array(self.ksout[-1]).reshape(-1,1)]
        for i in range(self.size):
            self.zs.append(np.array([self.biases[i][jindex] + j for jindex, j in enumerate(self.weights[i].dot(self.aas[i]))]))#self.weights[i].dot(self.aas[i]) + self.biases[i])
            if i < self.size - 1:
                self.aas.append(self.sigmoid(self.zs[i]))
            else:
                self.aas.append(self.sigmoid(self.zs[i]))
    
    def backward(self, y):
        print(np.array(self.ksout[0]).shape)
        self.dws = [0]*self.size
        self.dbs = [0]*self.size
        for i in range(self.size-1,-1,-1):
            if i == self.size - 1: #Wt*dL/dY*dactive
                self.dbs[i] = 2 * (self.aas[i+1] - y) * self.sigmoidd(self.zs[i]) #dbs also stores dx
            else:
                self.dbs[i] = self.weights[i+1].T.dot(self.dbs[i+1])*self.sigmoidd(self.zs[i])
            self.dws[i] = self.dbs[i].dot(np.array(self.aas[i]).T) #dL/dY*Xt (dactivate already factored in)

        #now time for the convolutional layers
        self.dks = []
        #dLdY is also dLdX
        dLdY = np.array(self.weights[0].T.dot(self.dbs[0])).reshape(self.convarch[-1][0], -1, self.outsizes[-1][0],self.outsizes[-1][1])
        for ln in range(len(self.kernels)-1,-1,-1): #for every layer, to backprop
            dl = [] #stores dks for layer
            dd = {} #stores dxs for layer, use dictionary because does the for loops but opposite
            for ki, kernel in enumerate(dLdY): #for every kernel that acted on the last inputs
                gradient = np.zeros(kernel[0].shape)
                for ii, inp in enumerate(kernel): #for every input that was acted on by the kernels
                    p = self.unpool(dLdY[ki][ii], self.convarch[ln][2], self.ps[ln][ki][ii], self.poutsizes[ln][0], self.poutsizes[ln][1])
                    z = p*self.backz[ln][ki][ii]
                    gradient += self.validCrossCorr(self.ksout[ln][ki][ii], z, pp=True) #gradient is sum of all of them
                    if ln != 0: #no point if last layer
                        #compute new dLdXs
                        dd[ii] = dd.get(ii, np.zeros((self.outsizes[ln-1][0], self.outsizes[ln-1][1]))) + self.fullConvolve(z, self.kernels)
                dl.append(gradient)
            self.dks.append(dl)
            if ln != 0: #sets new dxs
                dLdY = np.array(list(dd.values())).reshape(self.convarch[ln-1][0], -1, self.outsizes[ln-1][0], self.outsizes[ln-1][1])
                

    def update(self, lr):
        for i in range(self.size):
            self.weights[i] -= self.dws[i] * lr
            self.biases[i] -= self.dbs[i] * lr
        for index, i in enumerate(self.kernels):
            for jindex, j in enumerate(i):
                for kindex, k in enumerate(j):
                    for lindex, l in enumerate(k):
                        l -= self.dks[index][jindex][kindex][lindex]

    def descend(self, x, y, lr):
        self.forward(np.array(x).reshape(28, 28))
        self.backward(self.oneHotter(y))
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

#[[amount of kernels, kernel size, pool size]]
nn = ConvolutionalNeuralNetwork([28,28],[[16,3,2]],[64,32,10])
nn.train(xTrain[:10], yTrain[:10], 0.01, 200)
#print(str(nn.test(xTest,yTest,100)*100)+"%")