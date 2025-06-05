import matplotlib.pyplot as plt 
import numpy as np

barWidth = 0.25
fig = plt.subplots(figsize = (12, 8)) 

# kernels1 =     ['3e-4'] 
accuracy1 = [39.9982, 52.27, 44.544, 33.632] # 

# kernels2 =     ['6e-4'] 
accuracy2 = [36.36, 50.908, 41.816, 39.544] #   

br1 = np.arange(len(accuracy1)) 
br2 = [x + barWidth for x in br1] 


plt.bar(br1, accuracy1, color = 'r', width = barWidth, edgecolor = 'black', label = 'lr = 3e-4') 
plt.bar(br2, accuracy2, color = 'g', width = barWidth, edgecolor = 'black', label = 'lr = 6e-4') 


plt.xlabel('# of Conv. 2D and ReLU Layers', fontweight = 'bold', fontsize = 15) 
plt.ylabel('Average Final Validation Accuracy', fontweight = 'bold', fontsize = 15) 
plt.title("The Effect of the # of Conv. 2D and ReLU Layers", fontweight = 'bold', fontsize = 25) 

plt.xticks([r + barWidth - 0.125 for r in range(len(accuracy1))], ['1', '2', '3', '4']) 

plt.legend()
plt.show() 

