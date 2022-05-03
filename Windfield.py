import numpy as np
import imutils
import matplotlib.pyplot as plt
from gym import spaces, Env


class WindField(Env):
    
    direction={ 'N':0,'NE':1,'E':2,'SE':3,'S':4,'SW':5,'W':6,'NW':7 }
    change_coordinates={ 
                        'N':np.array([0,1,0]),
                        'NE':np.array([1,1,0]),
                        'E':np.array([1,0,0]),
                        'SE':np.array([1,-1,0]),
                        'S':np.array([0,-1,0]),
                        'SW':np.array([-1,-1,0]),
                        'W':np.array([-1,0,0]),
                        'NW':np.array([-1,1,0])
                        }
    
    
    def __init__(self,size):
        super(WindField,self).__init__()
        
        self.action_space=spaces.Discrete(8)
        
        self.observation_space=spaces.Tuple((spaces.Discrete(size),spaces.Discrete(size),spaces.Discrete(8)))
        
        self.state=(0,0,3)
        self.history=[(0,0)]
        
        self.targets=[]
        for i in range(2,5):
            self.targets.append((size-1,size-1)+(i,))
        
         
    def reset(self):
        
        self.state=(0,0,3)
        
        return np.array(self.state)

    
    def step(self,action):
        #self.history.append[self.state[:,1]]
        #return [self.state,reward,done]
        pass
    
    
    def render(self):
        canvas=np.zeros((100*self.size,100*self.size,3),dtype=np.uint8)
        
        canvas[0:100,0:100,:]=255*np.ones_like(canvas[0:100,0:100,:])
        
        canvas[(100*self.size)-100:100*self.size,(100*self.size)-100:100*self.size,1]=255*np.ones_like(canvas[0:100,0:100,1])
        
        airplane=imutils.resize(255*plt.imread(r"assets\airplane.png")[:,:,:3],width=100)
        pos=(100*self.state[1],100*self.state[0])
        facing=self.state[2]
        facing_img={
                    0:airplane,
                    1:imutils.rotate(airplane,angle=360-45),
                    2:imutils.rotate(airplane,angle=360-90),
                    3:imutils.rotate(airplane,angle=360-135),
                    4:imutils.rotate(airplane,angle=360-180),
                    5:imutils.rotate(airplane,angle=360-225),
                    6:imutils.rotate(airplane,angle=360-270),
                    7:imutils.rotate(airplane,angle=360-315)
                    }
        canvas[pos[0]:100+pos[0],pos[1]:100+pos[1]]=facing_img[facing]
        plt.imshow(canvas)
        
        if len(self.history)>1:
            for i in range(len(self.history)-1):
                plt.plot([100*self.history[i][0]+50,100*self.history[i+1][0]+50],[100*self.history[i][1]+50,100*self.history[i+1][1]+50],marker="o",color="red", linewidth=1)
        
        plt.grid(color="white")
        plt.show()
        