import numpy as np

class AmyloidCanvas():
    def __init__(self,Side=1024,pBranch=0.01,SigMomentum=0.001):
        self.Side = Side
        self.grid = np.zeros((Side,Side))
        self.pBranch = pBranch
        self.SigMomentum = SigMomentum
    def Nucleate(self,N_Nuclei,pos=None,ang=None,momentum=None):
        if pos is None:
            pos = np.random.uniform(0,self.Side,size=(N_Nuclei,2))
        if ang is None:
            ang = np.random.uniform(-np.pi,np.pi,size=(N_Nuclei))
        if momentum is None:
            momentum = np.random.normal(0,self.SigMomentum,size=(N_Nuclei))
        self.NucleiXY = pos
        self.growing = self.NucleiXY
        self.growingDir = np.repeat(ang,2,axis=0)
        self.growingDir[::2] += np.pi
        self.momentum = np.repeat(momentum,2,axis=0)
        self.momentum[::2] *= -1
        self.Paint()
        self.growing = np.repeat(self.NucleiXY,2,axis=0)
        return self
    def Branch(self,BranchAng=np.pi/4.5):
        newBranches = np.random.uniform(0,1,size=self.growing.shape[0]) < self.pBranch
        newDirs = self.growingDir[newBranches]+BranchAng
        if not len(newDirs):
            return self
        newDirs[np.random.uniform(0,1,size=newDirs.shape) < 0.5] -= BranchAng*2
        newBranches = self.growing[newBranches]
        newMomentum = np.random.normal(0,self.SigMomentum,size=newDirs.shape)
        self.growing = np.concatenate([self.growing,newBranches],axis=0)
        self.growingDir = np.concatenate([self.growingDir,newDirs],axis=0)
        self.momentum = np.concatenate([self.momentum,newMomentum],axis=0)
        return self
    def Grow(self,withMomentum=True):
        if withMomentum:
            self.growingDir += self.momentum
        v = np.array([np.cos(self.growingDir),np.sin(self.growingDir)]).T
        self.growing += v
        self.Paint()
        return self
    def Paint(self,intensity=1.):
        gridPos = np.rint(self.growing)
        gridPos[gridPos < 0] = np.nan
        gridPos[gridPos > self.Side-1] = np.nan
        gridPos = gridPos[~np.any(np.isnan(gridPos),axis=1)]
        gridPos = np.array(gridPos,dtype=int).T
        self.grid[(gridPos[0],gridPos[1])] += intensity
        
        # Delete whatever has gone beyond a 20% margin
        outOfBounds = np.any(((self.growing < -self.Side*0.2) | (self.growing > self.Side * 1.2)),axis=1)
        if np.any(outOfBounds):
            self.growing = self.growing[~outOfBounds]
            self.growingDir = self.growingDir[~outOfBounds]
            self.momentum = self.momentum[~outOfBounds]
        return self
    def make_grid(self,N_Nuclei,N_Steps,withMomentum=True):
        self.Nucleate(N_Nuclei)
        cur = 0
        for i in range(N_Steps):
            perc = int(i*100/N_Steps)
            if perc != cur:
                print("\rMaking image: {}%".format(int(i*100/N_Steps)),end='')
                cur = perc
            if len(self.growing) > 10.**5:
                raise ValueError("Calculation explodes - reduce your branching probability!")
            self.Grow(withMomentum=withMomentum)
            self.Branch()        