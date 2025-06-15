import pickle
import os

# Load from a file
with open('run_stlcg-1-auto_20250613_112936.pkl', 'rb') as f:
    running_time, margin, time_list = pickle.load(f)

print('sovling run_stlcg-1-auto takes %.3f s'%(running_time))
path_time, box_time, sum_time = map(sum, zip(*time_list))

print(f"We use {len(time_list)} rounds.")
print("Sum of first elements:", path_time)
print("Sum of second elements:", box_time)
print("Sum of third elements:", sum_time)

print(f"The value of margin is {margin}.")
# with open('run_wall-1-assigned_20250521_183852.pkl', 'rb') as f:
#     running_time, margin = pickle.load(f)
#
# print('sovling run_stlcg-1 takes %.3f s'%(running_time))
print("============================================")
# with open('run_wall-1-auto_20250602_162901.pkl', 'rb') as f:
#     total_time, margin, time_list = pickle.load(f)
#
# print('sovling wall-1-auto takes %.3f s'%(total_time))
# path_time, box_time, sum_time = map(sum, zip(*time_list))
#
# print(f"We use {len(time_list)} rounds.")
# print("Sum of first elements:", path_time)
# print("Sum of second elements:", box_time)
# print("Sum of third elements:", sum_time)
#
# print(f"The value of margin is {margin}.")
#
#
# print("============================================")
# with open('run_wall-1-assigned_20250602_171348.pkl', 'rb') as f:
#     total_time, margin, time_list = pickle.load(f)
#
# print('sovling wall-1-assign takes %.3f s'%(total_time))
# path_time, box_time, sum_time = map(sum, zip(*time_list))
#
# print(f"We use {len(time_list)} rounds.")
# print("Sum of first elements:", path_time)
# print("Sum of second elements:", box_time)
# print("Sum of third elements:", sum_time)
#
# print(f"The value of margin is {margin}.")
# print(time_list)