import matplotlib.pyplot as plt 
import numpy as np


rate =     ['8e-5','9e-5','1e-4', '2e-4','3e-4','4e-4','5e-4','6e-4',  '7e-4',  '8e-4'] 
accuracy = [38.84, 43.94, 34.843, 41.63, 46.8198, 46.8198, 40.153, 36.364, 40.15, 44.694] 

# python main.py --lr 4e-4 --kernels 3 --hidden 10 --adding 0 # 47.73, 40.91
# python main.py --lr 4e-4 --kernels 3 --hidden 10 --adding 0 # 50

# python main.py --lr 5e-4 --kernels 3 --hidden 10 --adding 0 # 50
# python main.py --lr 5e-4 --kernels 3 --hidden 10 --adding 0 # 31.82

# python main.py --lr 7e-4 --kernels 3 --hidden 10 --adding 0 # 50
# python main.py --lr 7e-4 --kernels 3 --hidden 10 --adding 0 # 34.091

plt.bar(rate, accuracy, color = "maroon") 
plt.xlabel('Learning Rate', fontweight = 'bold', fontsize = 15) 
plt.ylabel('Average Final Validation Accuracy', fontweight = 'bold', fontsize = 15) 
plt.title("Accuracy vs. Learning Rate (Real Data)", fontweight = 'bold', fontsize = 25)


plt.show() 