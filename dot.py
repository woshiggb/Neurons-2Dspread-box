import numpy as np

np.random.seed(42)

import pickle

class NLP:
    def __init__(self,box,s=4):
        if (len(box)==0):
            raise Exception("The box not have big or small")
        self.box = box
        self.wbox = []
        self.s = s
        self.lend = len(box)
        for i in range(self.lend):
            b = []
            if (i-s>=0): #上
                b.append(self.rom())
            else:
                b.append(-1)
            if (i+s<self.lend): #下
                b.append(self.rom())
            else:
                b.append(-1)
            if (i%s-1>=0): #左
                b.append(self.rom())
            else:
                b.append(-1)
            if (i%s+1<s): #右
                b.append(self.rom())
            else:
                b.append(-1)
            self.wbox.append(b)
        self.inp = [[i,self.wbox[i]] for i in range(len(box)) if box[i]==2] #信息
        self.tran = [[i,self.wbox[i]] for i in range(len(box)) if box[i]==1]
        self.out = [[i,self.wbox[i]] for i in range(len(box)) if box[i]==3]
        self.log = []
        self.log_num = []
    
    def forward(self,cin,tr=False):
        box = [0 for i in range(len(self.box))]
        for i in range(len(self.inp)):
            box[self.inp[i][0]] = cin[i]
        self.ifd = {}
        for i in range(len(self.inp)):
            if (self.inp[i][1][0]!=-1 and self.inp[i][0]-self.s>=0): #上
                box[self.inp[i][0]-self.s]+=self.inp[i][1][0]*cin[i]
                if (self.box[self.inp[i][0]-self.s]!=0):
                    self.ifd[self.inp[i][0]-self.s] = -1
                    box = self.up(self.inp[i][0]-self.s,box,tr=tr)
            if (self.inp[i][1][1]!=-1 and self.inp[i][0]+self.s<self.lend): #下
                box[self.inp[i][0]+self.s]+=self.inp[i][1][1]*cin[i]
                if (self.box[self.inp[i][0]+self.s]!=0):
                    self.ifd[self.inp[i][0]+self.s] = -1
                    box = self.up(self.inp[i][0]+self.s,box,tr=tr)
            if (self.inp[i][1][2]!=-1 and self.inp[i][0]%self.s-1>=0): #左
                box[self.inp[i][0]-1]+=self.inp[i][1][2]*cin[i]
                if (self.box[self.inp[i][0]-1]!=0):
                    self.ifd[self.inp[i][0]-1] = -1
                    box = self.up(self.inp[i][0]-1,box,tr=tr)
            if (self.inp[i][1][3]!=-1 and self.inp[i][0]%self.s+1<self.s): #右
                box[self.inp[i][0]+1]+=self.inp[i][1][3]*cin[i]
                if (self.box[self.inp[i][0]+1]!=0):
                    self.ifd[self.inp[i][0]+1] = -1
                    box = self.up(self.inp[i][0]+1,box,tr=tr)
        self.log.append(box)
        #print(box[4])
        #print(box)
        self.log_num.append([box[i[0]] for i in self.out])

    def up(self,inp,ob,tr=False):
        #if (len([0 for i in self.out if ob[i[0]]!=0])==len(self.out)):
        #    print("GOTO")
        #    return ob
        if (tr):
            for i in range(len(ob)):
                if (i%self.s==0):
                    print()
                print(round(ob[i],2),end=" ")
            print()
        if (self.wbox[inp][0]!=-1 and inp-self.s>=0): #上
            if (inp-self.s not in self.ifd):
                ob[inp-self.s]+=self.wbox[inp][0]*ob[inp]
                if (self.box[inp-self.s]!=0):
                    self.ifd[inp-self.s] = -1
                    self.ifd[inp] = -1
                    ob = self.up(inp-self.s,ob,tr=tr)

        if (self.wbox[inp][1]!=-1 and inp+self.s<self.lend): #下
            if (inp+self.s not in self.ifd):
                ob[inp+self.s]+=self.wbox[inp][1]*ob[inp]
                if (self.box[inp+self.s]!=0):
                    self.ifd[inp+self.s] = -1
                    self.ifd[inp] = -1
                    ob = self.up(inp+self.s,ob,tr=tr)
        if (self.wbox[inp][2]!=-1 and inp%self.s-1>=0): #左
            if (inp-1 not in self.ifd):
                ob[inp-1]+=self.wbox[inp][2]*ob[inp]
                if (self.box[inp-1]!=0):
                    self.ifd[inp-1] = -1
                    self.ifd[inp] = -1
                    ob = self.up(inp-1,ob,tr=tr)
        if (self.wbox[inp][3]!=-1 and inp%self.s+1<self.s): #右
            if (inp+1 not in self.ifd):
                ob[inp+1]+=self.wbox[inp][3]*ob[inp]
                if (self.box[inp+1]!=0):
                    self.ifd[inp+1] = -1
                    self.ifd[inp] = -1
                    ob = self.up(inp+1,ob,tr=tr)
        return ob

    def rom(self):
        return np.random.rand()
        #return np.random.randint(1,200)/100
        #return 0 

    def backward(self,y,lr=1):
        self.lr = lr
        self.ifd = {}
        for t in range(len(self.out)):
            self.y = y[t]
            i = self.out[t]
            self.ry = self.log_num[-1][t]
            self.e = self.ex(self.log_num[-1][t]) #误差
            e = self.e
            nexts = self.get_next_idx(i[0],0)
            if (nexts!=None):
                self.wbox[i[0]][0]+=e*lr
                self.back(nexts,1)
            nexts = self.get_next_idx(i[0],1)
            if (nexts!=None):
                self.wbox[i[0]][1]+=e*lr
                self.back(nexts,0)
            nexts = self.get_next_idx(i[0],2)
            if (nexts!=None):
                self.wbox[i[0]][2]+=e*lr
                self.back(nexts,3)
            nexts = self.get_next_idx(i[0],3)
            if (nexts!=None):
                self.wbox[i[0]][3]+=e*lr
                self.back(nexts,2)

    def ex(self,a):
        return (self.max10-(self.y-a))*(self.y-a)/self.lencins

    def back(self,nexts,last):
        e = self.ex(self.ry)
        the = nexts
        nexts = self.get_next_idx(the,0)
        #print(e*self.lr)
        if (nexts!=None and nexts not in self.ifd):
            self.wbox[the][0]+=e*self.lr
            self.ifd[nexts] = 0
            self.ifd[the] = 0
            self.back(nexts,1)
        nexts = self.get_next_idx(the,1)
        if (nexts!=None and nexts not in self.ifd):
            self.wbox[the][1]+=e*self.lr
            self.ifd[nexts] = 0
            self.ifd[the] = 0
            self.back(nexts,0)
        nexts = self.get_next_idx(the,2)
        if (nexts!=None and nexts not in self.ifd):
            self.wbox[the][2]+=e*self.lr
            self.ifd[nexts] = 0
            self.ifd[the] = 0
            self.back(nexts,3)
        nexts = self.get_next_idx(the,3)
        if (nexts!=None and nexts not in self.ifd):
            self.wbox[the][3]+=e*self.lr
            self.ifd[nexts] = 0
            self.ifd[the] = 0
            self.back(nexts,2)
        return 0

    def kil(self): #清除内存
        self.log = []
        self.logbox = []

    def train_a(self,cin,out,ep,lr,cont=1):
        self.lencins = len(cin)
        self.max10 = max([self.find10(b) for b in cin[i]+out[i] for i in range(len(cin))])
        for u in range(ep):
            for i in range(len(cin)):
                self.forward(cin[i],tr=False)
                self.backward(out[i],lr)
                if ((i+1)%cont==0):
                    print(f"error:{(sum(out[i])-sum(self.log_num[-1]))*100}%")
                #self.log = []

    def train_b(self,cin,out,ep,lr,cont=1):
        self.lencins = len(cin)
        self.max10 = 1
        for i in range(len(cin)):
            for u in range(ep):
                self.forward(cin[i],tr=False)
                self.backward(out[i],lr)
                if ((i+1)%cont==0):
                    print(f"error:{self.e}")
                #self.log = []

    def find10(self,n):
        ta = 10
        while (True):
            ta/=10
            if (ta<n):
                break
        while (True):
            ta*=10
            if (ta>=n):
                return ta


    def get_next_idx(self, idx, direction):
        """
        根据方向获取下一个节点的索引
        方向编码：0上, 1下, 2左, 3右
        """
        if direction == 0 and idx - self.s >= 0:
            return idx - self.s
        elif direction == 1 and idx + self.s < self.lend:
            return idx + self.s
        elif direction == 2 and idx % self.s - 1 >= 0:
            return idx - 1
        elif direction == 3 and idx % self.s + 1 < self.s:
            return idx + 1
        return None

    def save(self,path):
        with open(path, 'wb') as file:  # 'wb'模式表示以二进制写模式打开文件
            pickle.dump(self, file)

    def load(self,path):
        with open(path, 'rb') as file:
            obj = pickle.load(file)
        self.__dict__.update(obj.__dict__)

#cin = [2,1,3,3]
#N = NLP(cin,4)
#x = [[0.88],[0.66]]
#y = [[i[0]-0.22,0] for i in x]
#x = [[114,514],[910,1919],[125,346]]

#N.train_b(x,[[x[i][0]+x[i][1]] for i in range(len(x))],lr=0.1,ep=15000)
#N.train_b(x,y,lr=0.01,ep=50000)
#N.save("./-22-num.pkl")
#N.load("./-22-num.pkl")
#print(N.wbox)
#N.forward([],tr=True)
#for i in range(len(N.log[-1])):
#    if (i%N.s==0):
#        print()
#    print(N.log[-1][i],end=" ")
#print()
#print(N.log_num[-1])
