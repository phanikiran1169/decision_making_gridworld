import matplotlib.pyplot as plt

# Data
x = [0.20, 0.30, 0.40, 0.50, 0.60]
y = [74.00, 67.00, 76.00, 69.00, 55.00]

# Plotting the data
plt.plot(x, y, linewidth=3)  # Set line thickness to 3

# Adding title and labels
# plt.title("Sample Plot")
plt.xlabel("Uncertainity in predator movement ")
plt.ylabel("Success rate of pursuit (%)")

# Display the plot without a legend
# plt.show()
plt.savefig("success_rate.png", dpi=300)