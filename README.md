# MPTrj Dataset Loader for GNN-based MLIPs

This script provides a tool for loading and preprocessing data from the MPTrj database for graph neural networks (GNNs), enabling training for machine-learning interatomic potentials (MLIPs). It is tailored for molecular dynamics (MD) data and integrates with `PyTorch Geometric` (`pyg`) for seamless graph representation.

---

## Features

- **MPTrj Dataset Parsing:**
  - Reads MPTrj data from a JSON file containing material properties such as structure, energy, forces, and stresses.
  - Parses structural data into graph representations using `pymatgen`.

- **Graph Construction:**
  - Constructs graphs using both radius-based (`radius_graph`) and k-Nearest Neighbor (`knn_graph`) algorithms.
  - Periodic boundary conditions (PBC) are incorporated into atom positions.

- **MD-ready Data:**
  - Prepares graph data suitable for MD simulations, including atomic features, lattice parameters, and physical properties.

- **Dataset & DataLoader Integration:**
  - Implements `torch.utils.data.Dataset` for MPTrj data.
  - Compatible with `pyg`'s `DataLoader` for batching and training.

- **Data Caching (Optional):**
  - Supports caching of preprocessed data for faster subsequent loading.

---

## Prerequisites

### Required Libraries

- `pymatgen`
- `torch`
- `torch_geometric`
- `tqdm` (optional, for progress bars)
- `pickle` (optional, for caching)

### Input Data Format (MPTrj JSON)

The script expects an MPTrj dataset in the following JSON format:
```json
{
  "mp_id": {
    "graph_id": {
      "structure": {...},
      "energy_per_atom": ...,
      "force": [...],
      "stress": [...],
      "magmom": ...
    }
  }
}
```

## Advanced Usage

### Customizing Graph Construction

You can easily customize the way graphs are constructed by modifying the `process_data` method in the `StructureJsonData` class. For example:

- **Change the Cutoff Radius:** Adjust the `r_cut` value to modify the radius for constructing the radius-based graph.
- **Add Edge Features:** You can calculate additional edge attributes, such as bond angles, distances, or periodic image information.
- **Incorporate Global Features:** Add properties like lattice energy, temperature, or external conditions as global features.

---

### Integration with GNN Models

The `pyg.Data` objects produced by the dataset can be directly used with Graph Neural Network (GNN) architectures. Below is an example of integrating this loader with a simple GNN model:

#### Define a Simple GNN Model

```python
from torch_geometric.nn import GCNConv

class SimpleGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return x

import torch
from torch_geometric.loader import DataLoader

# Dataset and DataLoader
dataset = StructureJsonData("path/to/MPtrj.json", r_cut=3.0, k=4)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# Model, Loss, and Optimizer
model = SimpleGNN(input_dim=1, hidden_dim=64, output_dim=1)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()

# Training
for epoch in range(10):
    for batch in dataloader:
        optimizer.zero_grad()
        output = model(batch.x, batch.edge_index)
        loss = criterion(output, batch.y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")
```
