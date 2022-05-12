import numpy as np

def value_iteration(env,theta,gamma):
    V=np.random.random(size=env.observation_space_shape)
    V[env.size-1,env.size-1]=np.zeros(shape=env.observation_space_shape[2])
    while True:
        delta=0
        for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                for k in range(V.shape[2]):
                    v=V[i,j,k]
                    max_exp_reward=0
                    for l in range(len(env.action_space)):
                        temp=0
                        trans=env.trasitions(state=(i,j,k),action=l)
                        for m in trans:
                            temp+=(m[2]*(m[1]+gamma*V[m[0]]))
                        if temp>max_exp_reward:
                            max_exp_reward=temp
                    V[i,j,k]=max_exp_reward
                    delta=max(delta,abs(v-V[i,j,k]))
        if delta<theta:
            break
        
    policy=np.empty_like(V)
    for i in range(V.shape[0]):
            for j in range(V.shape[1]):
                for k in range(V.shape[2]):
                    exp_reward=[]
                    for l in range(len(env.action_space)):
                        temp=0
                        trans=env.trasitions(state=(i,j,k),action=l)
                        for m in trans:
                            temp+=m[2]*(m[1]+gamma*V[m[0]])
                        exp_reward.append(temp)
                    policy[i,j,k]=np.argmax(exp_reward)
    
    return policy
  
            
def play_policy(env,policy):
        
        state=env.state
        count=0
        while True:
            action=policy[state[0],state[1],state[2]]
            next_state,_,done=env.step(action)
            state=next_state
            count+=1
            if done or count==30:
                break
            
        env.render()