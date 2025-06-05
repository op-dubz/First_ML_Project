import matplotlib.pyplot as plt 
import numpy as np

barWidth = 0.25
fig = plt.subplots(figsize = (12, 8)) 

# kernels1 =     ['3-4'] 
accuracy1 = [51.705, 47.7225, 55.67775, 50.57, 52.3425, 43.1825] # 

# kernels2 =     ['6e-4'] 
accuracy2 = [50.908, 37.49775, 35.2275, 36.36, 38.0675, 32.95] #   

br1 = np.arange(len(accuracy1)) 
br2 = [x + barWidth for x in br1] 


plt.bar(br1, accuracy1, color = 'r', width = barWidth, edgecolor = 'black', label = 'lr = 3e-4') 
plt.bar(br2, accuracy2, color = 'g', width = barWidth, edgecolor = 'black', label = 'lr = 6e-4') 


plt.xlabel('# of Channels', fontweight = 'bold', fontsize = 15) 
plt.ylabel('Average Final Validation Accuracy', fontweight = 'bold', fontsize = 15) 
plt.title("The Effect of the # of Channels", fontweight = 'bold', fontsize = 25)

plt.xticks([r + barWidth - 0.125 for r in range(len(accuracy1))], ['10', '20', '30', '40', '50', '60']) 

plt.legend()
plt.show() 

