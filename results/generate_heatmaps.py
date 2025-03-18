import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Folder containing the CSV result files
results_folder = "gridworld_1"

# Initialize variables to store predator (pursuer) and prey (evader) coordinates
evader_positions = []
pursuer_positions = []
pursuit_success_count = 0
total_runs = 0
grid_size = None

# Function to process the CSV files
def process_csv_files():
    global pursuit_success_count, total_runs, grid_size
    
    # Loop through all CSV files in the folder
    for filename in os.listdir(results_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(results_folder, filename)
            with open(file_path, 'r') as file:
                reader = csv.reader(file)
                header = next(reader)  # Skip header row
                
                # Extract grid size from the first row of the file (first row in each file is same)
                if grid_size is None:
                    first_row = next(reader)  # Read the second row to get grid size
                    grid_size = (int(first_row[1]), int(first_row[0]))  # (width, height)
                    file.seek(0)  # Rewind to the beginning of the file
                    next(reader)  # Skip header row again

                # Process each row and store positions
                for row in reader:
                    evader_x = int(row[3])
                    evader_y = int(row[2])
                    pursuer_x = int(row[5])
                    pursuer_y = int(row[4])
                    pursuit_success = row[-1].strip() == "True"
                    
                    # Store evader and pursuer positions (consider all rows)
                    evader_positions.append((evader_x, evader_y))
                    pursuer_positions.append((pursuer_x, pursuer_y))
                    
                    # Count the pursuit success
                    if pursuit_success:
                        pursuit_success_count += 1
            
            total_runs += 1        

# Function to generate heatmaps
def generate_heatmaps():
    # Convert lists to numpy arrays for easier processing
    evader_positions_array = np.array(evader_positions)
    pursuer_positions_array = np.array(pursuer_positions)
    
    # Create a 2D grid to count positions
    evader_heatmap = np.zeros((grid_size[1], grid_size[0]))  # width x height
    pursuer_heatmap = np.zeros((grid_size[1], grid_size[0])) 

    # Fill the heatmaps
    for evader_x, evader_y in evader_positions_array:
        evader_heatmap[evader_y, evader_x] += 1  # Increment heatmap at the evader's position
        
    for pursuer_x, pursuer_y in pursuer_positions_array:
        pursuer_heatmap[pursuer_y, pursuer_x] += 1  # Increment heatmap at the pursuer's position
    
    # Define the common range for color scale normalization
    # vmin = min(evader_heatmap.min(), pursuer_heatmap.min())
    # vmax = max(evader_heatmap.max(), pursuer_heatmap.max())

    # Plot the heatmaps
    plt.figure(figsize=(12, 6))
    
    # Evader Heatmap
    plt.subplot(1, 2, 1)
    sns.heatmap(evader_heatmap, cmap="Blues", annot=False, cbar=True)
    plt.title("Evader")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks([])
    
    # Pursuer Heatmap
    plt.subplot(1, 2, 2)
    sns.heatmap(pursuer_heatmap, cmap="Reds", annot=False, cbar=True)
    plt.title("Pursuer")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.gca().invert_yaxis()
    cbar = plt.gca().collections[0].colorbar
    cbar.set_ticks([])
    
    # Save the heatmaps as an image file
    plt.tight_layout()
    heatmap_file = os.path.join(results_folder, "heatmaps.png")
    plt.savefig(heatmap_file)
    plt.close()

# Function to calculate and display success rate
def display_success_rate():
    success_rate = pursuit_success_count / total_runs * 100 if total_runs > 0 else 0
    success_rate_text = f"Predator Success Rate: {success_rate:.2f}%"
    
    # Display success rate in the terminal
    print(success_rate_text)
    
    # Save success rate to a text file
    success_rate_file = os.path.join(results_folder, "success_rate.txt")
    with open(success_rate_file, 'w') as file:
        file.write(success_rate_text)

# Main execution
def main():
    # Process all CSV files and extract coordinates
    process_csv_files()

    # Generate heatmaps for the evader and pursuer
    generate_heatmaps()

    # Calculate and display the success rate
    display_success_rate()

if __name__ == "__main__":
    main()
