import pickle

with open('run_wall-1_assigned_multi_20250608_194100.pkl', 'rb') as f4:
    loaded_data4, PWLs5= pickle.load(f4)

print('sovling run_wall-1_assigned for 4 agents takes %.3f s'%(loaded_data4))

print("===============================================================")

with open('run_wall-1_assigned_multi_20250608_194218.pkl', 'rb') as f5:
    loaded_data5, PWLs5= pickle.load(f5)

print('sovling run_wall-1_assigned for 5 agents takes %.3f s'%(loaded_data5))

print("===============================================================")

with open('run_wall-1_assigned_multi_20250608_194820.pkl', 'rb') as f6:
    loaded_data6, PWLs6= pickle.load(f6)

print('sovling run_wall-1_assigned for 6 agents takes %.3f s'%(loaded_data6))