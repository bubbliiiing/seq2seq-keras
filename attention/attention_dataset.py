import numpy as np

def get_data_recurrent(n, time_steps, input_dim, attention_column=2):
    x = np.random.normal(loc=0, scale=10, size=(n, time_steps, input_dim))
    y = np.random.randint(low=0, high=2, size=(n, 1))
    x[:, attention_column, :] = np.tile(y[:], (1, input_dim))
    return x, y 

x, y = get_data_recurrent(1, 10, 2)
print("x =",x)
print("y =",y)