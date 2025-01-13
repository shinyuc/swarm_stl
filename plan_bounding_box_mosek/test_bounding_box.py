import pickle

with open('../plan_waypoints_v1/PWLs.pkl', 'rb') as file:
    PWLs = pickle.load(file)

print(PWLs)
print(PWLs[0])
print(PWLs[0][0])