import numpy as np
import math
import random
from tqdm import tqdm
import time
import torch
import pandas as pd
from datasets import load_dataset
from torchvision.transforms import Compose, ToTensor, Resize, CenterCrop, Normalize
import timm


VECTOR_LENGTH = 24  # Length of the binary vector (12 for do_attns, 12 for do_mlps)
img_size = 32
mean, std = [0.5000, 0.5000, 0.5000], [0.5000, 0.5000, 0.5000]

transform = Compose([
    ToTensor(),
    Resize((img_size, img_size)),
    CenterCrop((img_size, img_size)),
    Normalize(mean=torch.tensor(mean), std=torch.tensor(std)),
])

cifar = load_dataset("cifar100")
imgs, labels = [], []
for sample in cifar["test"]:
    image = transform(sample["img"].convert("RGB"))
    imgs.append(image)
    labels.append(sample["fine_label"])

all_images = torch.stack(imgs)[:100]  
all_labels = torch.tensor(labels)[:100]

# Load pre-trained ViT model
vit = timm.create_model('vit_tiny_patch16_224.augreg_in21k_ft_in1k', img_size=32, num_classes=100, pretrained=True).cuda()
vit.load_state_dict(torch.load('vit_tiny_cifar100.pt'))
vit = vit.eval()


def accuracy_top_1(y_preds, y_true):
    """Calculate Top-1 accuracy"""
    top_1 = sum(int(y_preds[idx] == y_true[idx]) for idx in range(len(y_true)))
    return top_1 / len(y_true)

def get_reward(do_attns, do_mlps, images, labels):
    """Calculate the reward (Top-1 accuracy) for the pruned model"""
    # Ensure the binary vector is valid
    assert torch.all((do_attns == 0) | (do_attns == 1)).item()
    assert torch.all((do_mlps == 0) | (do_mlps == 1)).item()
    
    images = images.cuda()
    with torch.no_grad():
        logits = vit(images, do_attns, do_mlps)  # Forward pass through the model
        preds = torch.argmax(logits, dim=1)      # Get Top-1 predictions

    reward = accuracy_top_1(preds.cpu(), labels)
    return reward

# ============================================
# Simulated Annealing Implementation
# ============================================

def initialize_solution():
    """Generate an initial solution (random binary vector)"""
    return np.random.randint(2, size=VECTOR_LENGTH)

def generate_neighbor(current_solution):
    """Generate a neighboring solution by flipping one random bit"""
    neighbor = current_solution.copy()
    index = np.random.randint(len(neighbor))  
    neighbor[index] = 1 - neighbor[index]  
    return neighbor

def fitness(solution):
    """Calculate the fitness (Top-1 reward)"""
    binary_vector = torch.tensor(solution)
    do_attns = binary_vector[:12]
    do_mlps = binary_vector[12:]
    images = all_images
    labels = all_labels
    return get_reward(do_attns, do_mlps, images, labels)

def acceptance_probability(old_fitness, new_fitness, temperature):
    """Calculate the probability of accepting a new solution"""
    if new_fitness > old_fitness:
        return 1.0  # Always accept better solutions
    else:
        return math.exp((new_fitness - old_fitness) / temperature)

def update_temperature(initial_temperature, iteration, decay_rate):
    """Update the temperature as the iterations progress"""
    return initial_temperature * (decay_rate ** iteration)

def simulated_annealing(initial_temperature, decay_rate, max_iterations=1000, log_type="progress_bar"):
    """Main logic for simulated annealing with two logging options"""
    current_solution = initialize_solution()
    current_fitness = fitness(current_solution)
    best_solution = current_solution
    best_fitness = current_fitness
    temperature = initial_temperature

    # Statistics
    fitness_results = []  
    total_time = 0    
    time_per_iteration = []  

    if log_type == "progress_bar":
        iterator = tqdm(range(max_iterations), desc="Running Simulated Annealing")
    else:
        iterator = range(max_iterations)

    for iteration in iterator:
        start_time = time.time()

        neighbor = generate_neighbor(current_solution)
        neighbor_fitness = fitness(neighbor)

        if acceptance_probability(current_fitness, neighbor_fitness, temperature) > random.random():
            current_solution = neighbor
            current_fitness = neighbor_fitness

        if current_fitness > best_fitness:
            best_solution = current_solution
            best_fitness = current_fitness

        fitness_results.append(current_fitness)

        end_time = time.time()
        iteration_time = end_time - start_time
        time_per_iteration.append(iteration_time)
        total_time += iteration_time

        temperature = update_temperature(initial_temperature, iteration, decay_rate)

        if log_type == "detailed_log":
            print(f"Iteration {iteration + 1}/{max_iterations}: "
                  f"Best Fitness = {best_fitness:.4f}, "
                  f"Current Fitness = {current_fitness:.4f}, "
                  f"Temperature = {temperature:.4f}, "
                  f"Time = {iteration_time:.4f}s")

    greater_than_50 = sum(1 for f in fitness_results if f > 0.5)
    greater_than_75 = sum(1 for f in fitness_results if f > 0.75)

    return best_solution, best_fitness, greater_than_50, greater_than_75, total_time, time_per_iteration

# ============================================
# Parameter Settings and Experiments
# ============================================

initial_temperatures = [100, 50, 10, 1]  # Settings for initial temperature
cooling_rates = [0.99, 0.95, 0.90]       # Settings for cooling rate
max_iterations = 1000                    # Number of iterations

results = []

for initial_temperature in initial_temperatures:
    for decay_rate in cooling_rates:
        print(f"Running Simulated Annealing with Initial Temperature={initial_temperature}, Cooling Rate={decay_rate}")
        
        best_solution, best_fitness, greater_than_50, greater_than_75, total_time, time_per_iteration = simulated_annealing(
            initial_temperature=initial_temperature,
            decay_rate=decay_rate,
            max_iterations=max_iterations,
            log_type="progress_bar"  # Choose "progress_bar" or "detailed_log"
        )
        
        # Record results
        avg_time_per_iteration = np.mean(time_per_iteration)
        results.append({
            "Initial Temperature": initial_temperature,
            "Cooling Rate": decay_rate,
            "Solutions > 50%": greater_than_50,
            "Solutions > 75%": greater_than_75,
            "Total Time (mins)": total_time / 60,
            "Avg Time per Iteration (s)": avg_time_per_iteration
        })

# Print results as a DataFrame
df = pd.DataFrame(results)
print(df)