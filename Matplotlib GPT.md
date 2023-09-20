# Matplotlib GPT

### Installation:
`pip install matplotlib`


### Importing Matplotlib:
`import matplotlib.pyplot as plt`

### Creating a Simple Plot:
```
# Sample data
x = [1,2,3,4,5]
y = [2,4,6,8,10]

#Create a simple line plot
plt.plot(x,y)

#Display the plot
plt.show()
```

### Adding Labels and Titles:
```
plt.plot(x, y)
plt.xlabel('X-axis Label')
plt.ylabel('Y-axis Label')
plt.title('Simple Line Plot')
plt.show()
```

### Saving Plot:
`plt.savefig('plot.png')`
