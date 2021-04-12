import matplotlib.pyplot as plt
import csv

t=[]
x=[]
y=[]
z=[]

with open('output/result_1.csv') as file:
  reader = csv.reader(file,delimiter=',')
  next(reader)  
  for row in reader:
    t.append(float(row[0]))
    x.append(float(row[1]))
    y.append(float(row[2]))
    z.append(float(row[3]))

fig,axs = plt.subplots(2,2)

axs[0,0].plot(y,x)
axs[0,0].set_title('y-x')

axs[0,1].plot(x,z)
axs[0,1].set_title('x-z')

axs[1,0].plot(y,z)
axs[1,0].set_title('y-z')

plt.show()
