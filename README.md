# Multi-Armed-Bandit-Based-Client-Scheduling-for-Federated-Learning
Code Implemntion from the article Multi-Armed Bandit Based Client Scheduling for Federated Learning: 

short description : Aiming to minimize the time consumption of FL(Federated-Learning) training, this work considered the CS(Client-Scheduling) problem  both in the ideal scenario and non-ideal scenarios. For the ideal scenario, the writers proposed  the CS-UCB  algorithm  and  also  derived an upper bound of its performance regret. The upperbound suggests  that the performance regret of the proposed CS-UCB algorithm grows in a logarithmic way over communication rounds. However, the local datasets of clients are non-i.i.d. and  unbalanced and the availability of clients is dynamic in the non-ideal scenario. Thus, the writers introduced the fairness constraint to ensure each client could participate in a certain proportionof the communication rounds during the training process. The writers also proposed the CS-UCB-Qalgorithm based on UCB policy and virtual queue technique and provided an upper bound which shows that the performance regret of the proposed CS-UCB-Q algorithm has a sub-linear growthover  communication  rounds.


I implemented the algorithims and plot the graphs: 

For the ideal scenario:
![231312312313](https://user-images.githubusercontent.com/72392859/112658060-167b8680-8e64-11eb-92f1-5637281be68f.png)

For the non-ideal scenario:
![213123123123](https://user-images.githubusercontent.com/72392859/112658181-37dc7280-8e64-11eb-9e7c-58aa22329d4e.png)
