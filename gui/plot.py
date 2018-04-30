import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt
from matplotlib import animation as animation
from matplotlib import dates as mdates
from matplotlib.dates import  DateFormatter
import numpy as np

input_file = "tensor_output.csv"
plt.style.use('ggplot')

# set up for graph animation
fig = plt.figure() 
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    datetime, y = np.loadtxt(open(input_file, 'r'), delimiter=',', unpack=True,
        converters={ 0: mdates.bytespdate2num('%Y-%m-%d %H:%M:%S EST')}, usecols=[0, 3])

    ax1.clear()
    ax1.plot_date(datetime, y, 'g-')
    ax1.xaxis_date()
    ax1.xaxis.set_major_formatter( DateFormatter('%Y-%m-%d\n%H:%M:%S') )
    plt.subplots_adjust(bottom=.3)

    locs, labels = plt.xticks()
    plt.setp(labels, rotation=70)
    plt.title("Pedestrian Data")
    plt.ylabel("# of Pedestrians In Park")
    plt.xlabel("Date Time")

    plt.tight_layout()
    plt.grid(True)

ani = animation.FuncAnimation(fig, animate, interval=500)
plt.show()