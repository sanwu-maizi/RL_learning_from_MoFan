import numpy as np

x = np.array([[1,2,3,4],[6,7,8,9]])

x_add1=x[:,np.newaxis]
x_add2=x_add1[:,np.newaxis]
x_add3=x_add2[:,:,:,:,np.newaxis]

print(x_add1)
print(x_add1.shape)

print("")

print(x_add2)
print(x_add2.shape)

print("")

print(x_add3)
print(x_add3.shape)