import numpy as np
import matplotlib.pyplot as plt
from data import function_executions
from prison_strategies import *
import random 

# Simulation Parameters
simulation_duration = 24  # hours
num_developers = 100000



phi = 0.0000167  # per GB-s
phi_prime = 0.0000083  # per GB-s

mu_q = 0.000006  # per GB-s
mu_q_prime = 0.000003  # per GB-s

xq = 0.00003  # per GB-s
xq_prime = 0.000017  # per GB-s


def  client_payoff(client_action , service_provider_action):
   if client_action=='C' and service_provider_action=='C':
        return xq - phi 
   if client_action=='D' and service_provider_action=='C':
        return xq - phi_prime 
   if client_action=='C' and service_provider_action=='D':
        return xq_prime - phi 
   if client_action=='D' and service_provider_action=='D':
        return xq_prime - phi_prime 

def  serviceprovider_payoff(client_action , service_provider_action):
   if client_action=='C' and service_provider_action=='C':
        return phi - mu_q
   if client_action=='D' and service_provider_action=='C':
        return phi_prime - mu_q
   if client_action=='C' and service_provider_action=='D':
        return phi - mu_q_prime
   if client_action=='D' and service_provider_action=='D':
        return phi_prime - mu_q_prime
        
# Game Theory Strategies
def  always_cooperate(prev_opponent_action): 
     return 'C'

def  always_defect(prev_opponent_action): 
     return 'D'

def  random_strategy(prev_opponent_action): 
      return random.choice(['C', 'D'])

def tit_for_tat(prev_opponent_action):
    return prev_opponent_action if prev_opponent_action is not None else 'C'

def generous_tit_for_tat(prev_opponent_action):
    return 'C' if prev_opponent_action is None or random.random() < 0.9 else 'D'

def grim_trigger(history):
    return 'C' if 'D' not in history else 'D'

def pavlov(prev_outcome):
    return 'C' if prev_outcome else 'D'

def win_stay_lose_shift(prev_outcome):
    return 'C' if prev_outcome else 'D'

def play(strategy_func , strategy_name , prev_opponent_action ,history,prev_outcome  ):
    if strategy_name in ['GT']:
        return strategy_func(history)
        
    if strategy_name in ['PV','WTLS']:
        return strategy_func(prev_outcome)
        
    return  strategy_func(prev_opponent_action)   
    
    
        
                


# Run simulation
strategies = {'RNDM':random_strategy ,'ALWC' :always_cooperate ,'ALWD': always_defect,   'TFT': tit_for_tat, 'GTFT': generous_tit_for_tat, 'GT': grim_trigger, 'PV': pavlov, 'WTLS': win_stay_lose_shift}
markers = ['o', 's', '^', 'D', 'x', '*', 'p', 'h', '+', 'v', '<', '>']
client_result_matrix = np.zeros((len(strategies), len(strategies)))
serviceprovider_result_matrix = np.zeros((len(strategies), len(strategies)))

for i, (serviceprovider_strategy_name, serviceprovider_strategy_func) in enumerate(strategies.items()):
    for j, (client_strategy_name, client_strategy_func) in enumerate(strategies.items()):
        
        total_client_payoff , total_serviceprovider_payoff = 0, 0
        prev_client_action , prev_serviceprovider_action, prev_client_outcome ,prev_serviceprovider_outcome = None, None, True ,True
        client_history = []
        serviceprovider_history = []
        
        for hour in range(simulation_duration):
            hourly_demand =np.sum( function_executions[hour])
             
            # Determine strategy-based cooperation
            client_action = play(client_strategy_func , client_strategy_name , prev_serviceprovider_action ,serviceprovider_history,prev_client_outcome  )
            serviceprovider_action = play(serviceprovider_strategy_func , serviceprovider_strategy_name , prev_client_action ,client_history,prev_serviceprovider_outcome  )
            
            
            
            this_client_payoff = client_payoff(client_action , serviceprovider_action)
            this_serviceprovider_payoff = serviceprovider_payoff(client_action , serviceprovider_action)
            
            total_client_payoff += num_developers * this_client_payoff
            total_serviceprovider_payoff += num_developers  *  this_serviceprovider_payoff
            
            # Update for next iteration
            prev_client_action = client_action
            prev_serviceprovider_action = serviceprovider_action
            prev_client_outcome = this_client_payoff > 0
            prev_serviceprovider_outcome = this_serviceprovider_payoff > 0
            client_history.append(client_action)
            serviceprovider_history.append(serviceprovider_action)
            
        client_result_matrix[i,j] = total_client_payoff ;
        serviceprovider_result_matrix[i,j] = total_serviceprovider_payoff ;
    print(client_result_matrix) ;
    print(serviceprovider_result_matrix) ;
        #print(serviceprovider_strategy_name,client_strategy_name ,total_serviceprovider_payoff, total_client_payoff  )
        #TODO : Here need to plot result for each strategy


# تنظیمات اولیه نمودار





# Define colors and markers
colors = plt.cm.tab10(np.linspace(0, 1, len(strategies)))
marker_styles = ['o', 's', '^', 'D', 'p', '*', 'h', 'v', '<', '>']

# Create figure with 4×4 subplots
fig, axes = plt.subplots(nrows=4, ncols=4, figsize=(15, 12))  

# Adjust layout for better visibility
plt.subplots_adjust(hspace=0.5, wspace=0.3)

# Plot each strategy
for i, (serviceprovider_strategy_name, _) in enumerate(strategies.items()):
    row, col = divmod(i, 2)  # Convert index to (row, col) format
    ax1 = axes[row, col * 2]  # Left: Service Provider Payoff
    ax2 = axes[row, col * 2 + 1]  # Right: Client Payoffs

    index = np.arange(len(strategies))
    serviceprovider_payoffs = serviceprovider_result_matrix[i, :]
    client_payoffs = client_result_matrix[i, :]

    # Bar plot for Service Provider
    ax1.bar(index, serviceprovider_payoffs, color=colors[i], label=serviceprovider_strategy_name)
    ax1.set_xticks(index)
    ax1.set_xticklabels(strategies.keys(), rotation=45, fontsize=8)
    ax1.set_ylabel('Payoff', fontsize=9)
    ax1.set_title(f'Service Provider: {serviceprovider_strategy_name}', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.legend(fontsize=8)

    # Line plot for Client Payoffs
    ax2.plot(index, client_payoffs, marker=marker_styles[i % len(marker_styles)], 
             color=colors[i], linewidth=2, markersize=6, label=f'Client vs {serviceprovider_strategy_name}')
    ax2.set_xticks(index)
    ax2.set_xticklabels(strategies.keys(), rotation=45, fontsize=8)
    ax2.set_ylabel('Payoff', fontsize=9)
    ax2.set_title(f'Client Strategies vs {serviceprovider_strategy_name}', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.legend(fontsize=8)

# Show the plot
plt.show()



# Create index for strategies
strategy_names = list(strategies.keys())
x = np.arange(len(strategy_names))  # X-axis (Service Provider Strategies)
y = np.arange(len(strategy_names))  # Y-axis (Client Strategies)
X, Y = np.meshgrid(x, y)  # Create meshgrid

# Get Payoff Matrices
Z_provider = serviceprovider_result_matrix  # Service Provider Payoff
Z_client = client_result_matrix  # Client Payoff

# Create 3D Figure
fig = plt.figure(figsize=(14, 6))

# Plot Service Provider Payoff
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z_provider, cmap='viridis', edgecolor='k', alpha=0.8)
ax1.set_title("Service Provider Payoff Matrix", fontsize=8)
ax1.set_xlabel("Service Provider Strategy", fontsize=8)
ax1.set_ylabel("Client Strategy", fontsize=8)
ax1.set_zlabel("Payoff", fontsize=8)
ax1.set_xticks(x)
ax1.set_xticklabels(strategy_names, rotation=45, fontsize=6)
ax1.set_yticks(y)
ax1.set_yticklabels(strategy_names, rotation=-45, fontsize=6)

# Plot Client Payoff
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z_client, cmap='coolwarm', edgecolor='k', alpha=0.8)
ax2.set_title("Client Payoff Matrix", fontsize=8)
ax2.set_xlabel("Service Provider Strategy", fontsize=8)
ax2.set_ylabel("Client Strategy", fontsize=8)
ax2.set_zlabel("Payoff", fontsize=8)
ax2.set_xticks(x)
ax2.set_xticklabels(strategy_names, rotation=45, fontsize=6)
ax2.set_yticks(y)
ax2.set_yticklabels(strategy_names, rotation=-45, fontsize=6)

# Adjust layout
plt.tight_layout()
plt.show()