import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation as animation
from matplotlib import dates as mdates
import numpy as np


input_file = "tensor_output.csv"

# set up for graph animation
fig = plt.figure() 
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    datetime, y = np.loadtxt(open(input_file, 'r'), delimiter=',', unpack=True,
        converters={ 0: mdates.bytespdate2num('%Y-%m-%d %H:%M:%S EST')}, usecols=[0, 3])

    ax1.clear()
    ax1.plot_date(datetime, y, 'g-')

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=90)
    plt.title("Pedestrian History Data")
    plt.ylabel("# of Pedestrians in Park")
    plt.xlabel("Date Time")
    plt.tight_layout()
    plt.grid(True)

ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()