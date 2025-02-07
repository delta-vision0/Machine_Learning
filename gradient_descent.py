import numpy as np

x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,10])

w=0
b=0

learning_rate = 0.0001
num_iteration =10000

for i in range(num_iteration):
    y_pred=w*x+b
    error=y_pred-y
    
    dw = (2/len(x))*np.sum(error*x)
    db = (2/len(x))*np.sum(error)

    w=w-learning_rate*dw
    b=b-learning_rate*db

    if i%100==0:
        MSE=np.mean(error**2)
        print(f"iteration {i} :w={w:.4f} b={b:.4f} MSE = {MSE:.4f}")
        #plt.scatter(x,y,color="blue",label = "True Value")
