import matplotlib.pyplot as plt 
import numpy as np

barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

# kernels1 =     ['3e-4'] 
accuracy1 = [54.546, 39.5442, 52.27, 44.5442, 47.2722, 44.5442, 42.2742, 39.09, 40.906] # 

# kernels2 =     ['6e-4'] 
accuracy2 = [49.0902, 40.452, 50.908, 46.8162, 42.7242, 35.9084, 40.002, 41.3682, 37.2182] #   

br1 = np.arange(len(accuracy1)) 
br2 = [x + barWidth for x in br1]


plt.bar(br1, accuracy1, color = 'r', width = barWidth, edgecolor = 'black', label = 'lr = 3e-4') 
plt.bar(br2, accuracy2, color = 'g', width = barWidth, edgecolor = 'black', label = 'lr = 6e-4') 


plt.xlabel('Kernel Sizes', fontweight = 'bold', fontsize = 15) 
plt.ylabel('Average Final Validation Accuracy', fontweight = 'bold', fontsize = 15) 
plt.title("The Effect of Different Kernel Sizes", fontweight = 'bold', fontsize = 25)

plt.xticks([r + barWidth - 0.125 for r in range(len(accuracy1))], ['1', '2', '3', '4', '5', '6', '7', '8', '9'])

plt.legend()
plt.show() 


#run again to check validation of graph since max pool = 2, now instead of max pool = kernel_size