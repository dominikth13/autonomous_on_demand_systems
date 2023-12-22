import torch
import torch.nn as nn
import torch.optim as optim
from deep_rl_setup import NeuroNet , td_error , generate_driver_action_pairs_without_weights
import random
import time

from logger import LOGGER


initialization_first_time = True
# Define a simple neural network

# Initialize the network
net = NeuroNet()
# Initialize a new network
target_net = NeuroNet()

if initialization_first_time == False:
    net.load_state_dict(torch.load('net_state_dict.pth'))
    #either model.eval() or model.train() depending on what we currently doing
    target_net.load_state_dict(torch.load('target_net_state_dict.pth'))
    #either model.eval() or model.train() depending on what we currently doing



 # Commonly used for classification problems
optimizer = optim.SGD(net.parameters(), lr=0.01)  # Stochastic Gradient Descent

# Training loop
for epocch_target in range(100):
    LOGGER.info(epocch_target)
    original_state_dict = net.state_dict() # Save the state dict of the original network
    # Load the state dict of the original network into the new network
    target_net.load_state_dict(original_state_dict)
    # Ensure that the new network is in the same mode (train or eval) as the original
    target_net.train(mode=net.training)
      

    df = generate_driver_action_pairs_without_weights()

    for epoch in range(100):  # loop over the dataset multiple times
        index = random.randint(0,len(df)-1)

        optimizer.zero_grad()  # zero the parameter gradients
        output_target_net = target_net(df.loc[index,'Target Position'].lat, df.loc[index,'Target Position'].lon, df.loc[index,'Target Time'])
        # Forward pass
        output = net(df.loc[index,'Current Position'].lat, df.loc[index,'Current Position'].lon, df.loc[index,'Current Time'])

        # Compute loss
        loss = td_error(output, output_target_net, df.loc[index,'Reward'])
        #loss = loss.pow(2)  # Squaring the TD error (if needed) I don`t want negative losses

        # Backward pass
        loss.backward()

        # Optimize
        optimizer.step()

        # Print statistics
        print(f'Epoch {epoch + 1}, Loss: {loss.item()}')

print('Finished Training')

torch.save(net.state_dict(), 'net_state_dict.pth')
torch.save(target_net.state_dict(), 'target_net_state_dict.pth')
