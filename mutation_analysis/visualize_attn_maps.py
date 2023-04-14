import torch
import matplotlib.pyplot as plt
import pandas as pd
import os

# Load precomputed attn. maps and sequence labels
results = torch.load(f'mutation_analysis/attn_maps/precomputed_attn_maps_0.pt')
sequences = pd.read_excel("mutation_analysis/MOESM7_ESM.xlsx")['sequence']

# Plot attn. maps across all layers for sequence i (make sure correct .pt file is loaded)
i = 1
n, num_layers, num_heads, seq_length = results.shape[:4]
fig, axs = plt.subplots(4, num_heads//4, figsize=(10, 8))

for l in range(num_layers):
    for ax, head in zip(axs.flatten(), results[i % n, l]):
        # Crop <cls> and <eos> token from map before plotting
        ax.imshow(head[1:-1, 1:-1])
        ax.axis('off')

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.suptitle(f'Attn. Maps for {sequences[i]} (Layer {l+1} Heads 1-20)', fontsize=16)

    folder_name = f'mutation_analysis/attn_map_plots/seq_{i}'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    plt.savefig(os.path.join(folder_name, f'layer_{l+1}'))