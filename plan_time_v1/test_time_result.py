import pickle

with open('PWLs.pkl', 'rb') as file:
    PWLs = pickle.load(file)

print(PWLs)
print(PWLs[0])