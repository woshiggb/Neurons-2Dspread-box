### Neurons-2Dspread-box
Functions
NGF class:

forward - Takes input values cin.
Effect: Performs forward propagation.
You can check the return value through NGF.log_num.
up - A recursive helper for forward.
rom - Returns a random bias, used for initialization.
backward - Takes expected output y.
ex - Calculates the error.
back - A recursive helper for backward.
kil - Cleans up memory.
train_a - Takes sample input cin, sample output out, number of training epochs ep, and learning rate lr (which needs to increase as the value increases).
train_b - Similar to train_a, but generally yields better training results.
### Code Example
'''python
from dot import NGF

Where 0 cannot propagate but can be propagated to, 2 is input, 3 is output, both can have multiple instances.
cin = [2, 1, 1, 3,
       0, 1, 1, 1]
l = 4 # Number of items per row
N = NGF(cin, l) # Initialization (N.s=l, N.log_num is the output...)
x = [[0.1], [0.2]] # Setting inputs and outputs
y = [[0.1], [0.2]]
N.train_b(x, y, lr=0.1, ep=10000) # Training
N.save('./pkl') # Save model

N.load('./pkl')

print(N.wbox) # Print bias list
N.forward([0.2], tr=True) # Perform one forward propagation, tr indicates to print the process
for i in range(len(N.log[-1])):
    if (i % N.s == 0):
        print()
    print(N.log[-1][i], end=" ") # Print the box configuration
'''

This code snippet demonstrates how to initialize the NGF class with a specific structure (cin), train it with given inputs and expected outputs (x and y), save the trained model, load it again, and finally perform a forward propagation to see the output. The tr parameter in forward method is used to print the propagation process for debugging or understanding the flow.
