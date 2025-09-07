# POLICY EVALUATION

## AIM
To implement and evaluate a given policy in a reinforcement learning environment using the iterative policy evaluation algorithm, and to analyze the resulting state-value function for different policies.

## PROBLEM STATEMENT
In reinforcement learning, the objective is to estimate the value function of a policy, which represents the expected long-term reward for each state when following that policy. The problem is to implement policy evaluation, apply it to different policies in a given environment (e.g., FrozenLake), compute their state-value functions, and compare the performance of the policies based on their values.

## POLICY EVALUATION FUNCTION
```python
# Name : Dharshini K
# Register No : 212223230047

def policy_evaluation(pi, P, gamma=1.0, theta=1e-10):
    prev_V = np.zeros(len(P))
    while True:
        V = np.zeros(len(P))
        for s in range(len(P)):
            for prob, next_state, reward, done in P[s][pi(s)]:
                V[s] += prob * (reward + gamma * prev_V[next_state] * (not done))
        if np.max(np.abs(prev_V - V)) < theta:
            break
        prev_V = V.copy()
    return V
```

## OUTPUT
### First policy
<img width="849" height="623" alt="image" src="https://github.com/user-attachments/assets/f97a5131-7153-430f-819d-40b28cfab731" />
<img width="945" height="147" alt="image" src="https://github.com/user-attachments/assets/43b6bd72-b534-4515-8f54-07ad69f71260" />
<img width="698" height="240" alt="image" src="https://github.com/user-attachments/assets/afd773a9-5627-47cb-95e3-1019fa783239" />

### Second policy
<img width="753" height="728" alt="image" src="https://github.com/user-attachments/assets/eddd9ad3-eb30-4cf1-9d19-4aeacd1966f5" />
<img width="953" height="174" alt="image" src="https://github.com/user-attachments/assets/63e7c806-f348-468c-9814-2ae858f3e5cb" />
<img width="704" height="235" alt="image" src="https://github.com/user-attachments/assets/44b48643-54e2-4fa9-8a0e-06b852889427" />

### Comparing first and second
<img width="1103" height="370" alt="image" src="https://github.com/user-attachments/assets/7dfbc8af-68b2-4db0-ab7b-81cd59ffcf04" />

## RESULT
The policy evaluation experiment was successfully implemented. The state-value functions for two different policies were obtained and compared. The results show that the first policy performs better than the second policy, as it provides higher expected returns in most states.
