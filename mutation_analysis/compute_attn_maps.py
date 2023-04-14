import torch
import esm
import matplotlib.pyplot as plt
import pandas as pd

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

# Load data and convert to list of tuples, i.e. ('0000000000111', DVLTFNSAAYDKR)
df = pd.read_excel("mutation_analysis/MOESM7_ESM.xlsx")
data = list(df[['genotype', 'sequence']].to_records(index=False))

# Compute and store attention maps, grouping n samples for each file
n = 2048
for i in range(0, len(data), n):
    batch_labels, batch_strs, batch_tokens = batch_converter(data[i:i + n])
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    # Extract per-residue representations (on CPU)
    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)

    torch.save(results['attentions'], f'mutation_analysis/attn_maps/precomputed_attn_maps_{i//n}.pt')

# # Plot attn. maps for each layer given a sample
# num_layers, num_heads, seq_length = results['attentions'].shape[1:4]
# fig, axs = plt.subplots(4, 5, figsize=(10, 8))

# for l in range(num_layers):
#     for ax, head in zip(axs.flatten(), results['attentions'][0, l]):
#         ax.imshow(head)
#         ax.axis('off')

#     fig.subplots_adjust(hspace=0.3, wspace=0.3)
#     plt.suptitle(f'Attn. Maps for {seq} (Layer {l} Heads 1-20)', fontsize=16)
#     plt.savefig(f'mutation_analysis/attn_maps/layer_{l}')
