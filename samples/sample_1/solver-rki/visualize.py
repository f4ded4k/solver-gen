import matplotlib.pyplot as plt
import csv

t=[]
x=[]
y=[]

with open('output/result_1.csv') as file:
  reader = csv.reader(file,delimiter=',')
  next(reader)
  for row in reader:
    t.append(float(row[0]))
    x.append(float(row[1]))
    y.append(float(row[2]))

fig,axs = plt.subplots(2,2)

axs[0,0].plot(t,x)
axs[0,0].set_title('t-x')

axs[0,1].plot(t,y)
axs[0,1].set_title('t-y')

plt.show()
