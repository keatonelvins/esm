import torch
import esm
import matplotlib.pyplot as plt
import pandas as pd

# Load ESM-2 model
model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_5()
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
