import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0.0,2.0,10)
y3 = [3]*10
y1 = (-4*x+16)/5
y2 = (-8*x+20)/5

c_1 = (-4*x+ 13)/4
c_2 = (-5*x+ 14)/4
c_3 = (-x+ 3)
c_4 = (-3*x+ 7)/2

y4 = np.minimum(y1, y2)

y5 = np.minimum(y2, y3)

y7 = np.minimum(y4, y5)

c_y1 = np.minimum(y7, c_1)

c_y2 = np.minimum(c_y1, c_2)

c_y3 = np.minimum(c_y2, c_3)

c_y4 = np.minimum(c_y3, c_4)

plt.fill_between(x, 0, c_y4, where=c_y4<=c_y3,facecolor='grey', alpha=0.5)


plt.plot(x, y1, label=r'$4x+5y<=16$')
plt.plot(x, y2, label=r'$8x+5y<=20$')
plt.plot(x, y3, label=r'$0<=y<=3$')

plt.plot(x, c_1, c = 'red',label=r'$4x+4y<=13$')
plt.plot(x, c_2, c = 'red',label=r'$5x+4y<=14$')
plt.plot(x, c_3, c = 'red',label=r'$x+y<=3$')
plt.plot(x, c_4, c = 'red',label=r'$3x+2y<=7$')



plt.scatter(1.0,1.0,c = 'r')
plt.scatter(1.0,2.0,c = 'r')
plt.scatter(0.0,1.0,c = 'r')
plt.scatter(0.0,2.0,c = 'r')
plt.scatter(0.0,0.0,c = 'r')
plt.scatter(1.0,0.0,c = 'r')
plt.scatter(0.0,3.0,c = 'r')


plt.xlim((0.0, 2.0))
# plt.ylim((0.0, 3.5))

plt.xlabel(r'$x1$')
plt.ylabel(r'$x2$')
plt.legend(bbox_to_anchor=(1.05, 1),loc=2)
plt.show()