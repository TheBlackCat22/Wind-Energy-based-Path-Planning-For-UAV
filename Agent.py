import Windfield_Env

theta = 0.01
delta = 100000
actions = list(a.change_coordinates.keys())
while delta >= theta:
    delta = 0
    for s in S:
        v = V[s[0], s[1]]
        V_max = -1000
        for act in actions:
            probs = np.array(a.transition_prob(state = np.hstack((s, np.array([3]))), action = act, wind_field=WindField))[:,1]
            next_s = np.array(a.transition_prob(state = np.hstack((s, np.array([3]))), action = act, wind_field=WindField))[:,0]  
            reward = np.array([a.reward(state = sd, windfield=WindField) for sd in next_s])
            temp = (probs*reward).sum()
            V_max = np.array([temp, V_max]).max()
        V[s[0],s[1]] = V_max
        delta = np.array([delta, abs(v - V[s[0],s[1]])])