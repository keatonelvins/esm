import torch
import esm
import matplotlib.pyplot as plt

# Load ESM-2 model
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

seq = "DVLTFNSAAYNNK"
# Prepare data
data = [
    ("protein1", seq)
]
batch_labels, batch_strs, batch_tokens = batch_converter(data)
batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

# Extract per-residue representations (on CPU)
with torch.no_grad():
    results = model(batch_tokens, repr_layers=[33], return_contacts=True)

num_layers, num_heads, seq_length = results['attentions'].shape[1:4]
fig, axs = plt.subplots(4, 5, figsize=(10, 8))

for l in range(num_heads):
    for ax, head in zip(axs.flatten(), results['attentions'][0, l]):
        ax.imshow(head)
        ax.axis('off')

    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.suptitle(f'Attn. Maps for {seq} (Layer {l} Heads 1-20)', fontsize=16)
    plt.savefig(f'mutation_analysis/attn_maps/layer_{l}')

# # Look at the unsupervised self-attention map contact predictions
# for (_, seq), tokens_len, attention_contacts in zip(data, batch_lens, results["contacts"]):
#     plt.matshow(attention_contacts[: tokens_len, : tokens_len])
#     plt.title(f'Contact Prediction for {seq}')
#     plt.show()
