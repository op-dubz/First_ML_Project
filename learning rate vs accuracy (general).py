import matplotlib.pyplot as plt 
import numpy as np


rate =     ["1e-5", "2e-5","3e-5",'4e-5','5e-5', '6e-5', '7e-5', '8e-5','9e-5','1e-4', '2e-4','3e-4','4e-4','5e-4','6e-4',  '7e-4',  '8e-4', '9e-4', '1e-3'] 
accuracy = [34.091, 43.18, 38.636, 45.45, 45.45, 45.45,  45.45, 43.181, 45.45, 45.45,  44.7,  52.27, 47.27, 42.726, 50.908, 41.816, 38.632, 44.9992, 38.6372] 

# rate = [1e-4 / 2, 1e-4, 1e-3] 
# accuracy = [47.72, 41.2, 45.45]  

# kernel size = out_shape = in_shape = 3, pool size = 2, adding = 0, hidden_units = 10









#plt.bar(rate, accuracy, color = "maroon", width = 0.5) 
'''
plt.plot(rate, accuracy, color = "maroon") 
plt.xlabel('Learning Rate') 
plt.ylabel('Average Final Validation Accuracy') 
plt.title("Learning Rate vs. Accuracy")
'''
plt.bar(rate, accuracy, color = "maroon") 
plt.xlabel('Learning Rate', fontweight = 'bold', fontsize = 15) 
plt.ylabel('Average Final Validation Accuracy', fontweight = 'bold', fontsize = 15) 
plt.title("Accuracy vs. Learning Rate", fontweight = 'bold', fontsize = 25)

# plt.savefig("lraccuracy.png") 

plt.show() 