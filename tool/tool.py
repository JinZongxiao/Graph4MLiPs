# -*- coding = utf-8 -*-
# @Time : 2024/11/19 18:37
# @Author : Jin ZONGXIAO__BJUT
# @File : tool.py
# @Software : PyCharm
import json

from pymatgen.core.structure import Structure

import torch
from torch.utils.data import Dataset

from torch_geometric.data import Data
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.loader import DataLoader

# 从Json文件中读取数据
## usage

# 全局字典
atom_number = {
    "H": 1, "He": 2, "Li": 3, "Be": 4, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Ne": 10,
    "Na": 11, "Mg": 12, "Al": 13, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Ar": 18, "K": 19, "Ca": 20,
    "Sc": 21, "Ti": 22, "V": 23, "Cr": 24, "Mn": 25, "Fe": 26, "Co": 27, "Ni": 28, "Cu": 29, "Zn": 30,
    "Ga": 31, "Ge": 32, "As": 33, "Se": 34, "Br": 35, "Kr": 36, "Rb": 37, "Sr": 38, "Y": 39, "Zr": 40,
    "Nb": 41, "Mo": 42, "Tc": 43, "Ru": 44, "Rh": 45, "Pd": 46, "Ag": 47, "Cd": 48, "In": 49, "Sn": 50,
    "Sb": 51, "Te": 52, "I": 53, "Xe": 54, "Cs": 55, "Ba": 56, "La": 57, "Ce": 58, "Pr": 59, "Nd": 60,
    "Pm": 61, "Sm": 62, "Eu": 63, "Gd": 64, "Tb": 65, "Dy": 66, "Ho": 67, "Er": 68, "Tm": 69, "Yb": 70,
    "Lu": 71, "Hf": 72, "Ta": 73, "W": 74, "Re": 75, "Os": 76, "Ir": 77, "Pt": 78, "Au": 79, "Hg": 80,
    "Tl": 81, "Pb": 82, "Bi": 83, "Po": 84, "At": 85, "Rn": 86, "Fr": 87, "Ra": 88, "Ac": 89, "Th": 90,
    "Pa": 91, "U": 92, "Np": 93, "Pu": 94, "Am": 95, "Cm": 96, "Bk": 97, "Cf": 98, "Es": 99, "Fm": 100,
    "Md": 101, "No": 102, "Lr": 103, "Rf": 104, "Db": 105, "Sg": 106, "Bh": 107, "Hs": 108, "Mt": 109,
    "Ds": 110, "Rg": 111, "Cn": 112, "Fl": 114, "Lv": 116, "Ts": 117, "Og": 118
}

class StructureJsonData(Dataset):

    def __init__(self, mptrj_path: str, r_cut: float = 3.0, k: int = 4):
        self.mptrj_path = mptrj_path
        if mptrj_path.endswith(".json"):
            self.data = {}
            with open(mptrj_path, "r") as f:
                self.data = json.load(f)
        self.keys = [(mp_id, graph_id) for mp_id, dct in self.data.items() for graph_id in dct]
        #    mp_id         graph_id
        # [('mp-1005792', 'mp-1005792-0-1'),
        #  ('mp-1005792', 'mp-1005792-0-0'),
        #  ('mp-1005792', 'mp-1005792-1-1'),
        #  ('mp-1005792', 'mp-1005792-1-0')]
        self.r_cut = r_cut
        self.k = k

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx: int):
        return self.process_data(idx)

    def process_data(self, idx):
        mp_id, graph_id = self.keys[idx]
        struct = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
        lattice_abc = struct.lattice.abc
        lattice_angles = struct.lattice.angles
        lattice_volume = struct.lattice.volume
        atom_num = struct.num_sites
        # atom_elements_type = [spe.name for spe in struct.species]
        atom_element_number = [atom_number[spe.name] for spe in struct.species]
        coords = [site.coords for site in struct.sites]
        pbc_site = [site.frac_coords for site in struct.sites]

        energy = self.data[mp_id][graph_id]['energy_per_atom']
        force = self.data[mp_id][graph_id]['force']
        stress = self.data[mp_id][graph_id]['stress']
        stress_vaspFormat = [[x * (-0.1) for x in row] for row in stress]
        magmom = self.data[mp_id][graph_id]['magmom']

        x = torch.tensor(atom_element_number, dtype=torch.float32)
        y = torch.tensor([energy], dtype=torch.float32)
        pos = torch.tensor(coords, dtype=torch.float32)
        pbc_pos = torch.tensor(pbc_site, dtype=torch.float32)
        edge_index_radius = radius_graph(pos, r=self.r_cut)
        edge_index_knn = knn_graph(pos, k=self.k)

        atom_num = torch.tensor(atom_num, dtype=torch.float32)
        lattice_abc = torch.tensor(lattice_abc, dtype=torch.float32)
        lattice_angles = torch.tensor(lattice_angles, dtype=torch.float32)
        lattice_volume = torch.tensor(lattice_volume, dtype=torch.float32)

        force = torch.tensor(force, dtype=torch.float32)
        stress = torch.tensor(stress_vaspFormat, dtype=torch.float32)
        stress_vaspFormat = torch.tensor(stress_vaspFormat, dtype=torch.float32)
        if magmom == None:
            magmom = 0
        magmom = torch.tensor(magmom, dtype=torch.float32)

        graphData = Data(x=x, y=y, pos=pos, edge_index=edge_index_radius,
                         pbc_pos=pbc_pos, edge_index_radius=edge_index_radius, edge_index_knn=edge_index_knn,
                         atom_num=atom_num,
                         lattice_abc=lattice_abc, lattice_angles=lattice_angles, lattice_volume=lattice_volume,
                         force=force, stress=stress, stress_vaspFormat=stress_vaspFormat, magmom=magmom)
        return graphData

import os
import pickle
from tqdm import tqdm

cache_dir = "dataloader_cache"
os.makedirs(cache_dir, exist_ok=True)
mptrj_path = 'C:\\Users\\Thinkstation2\\Desktop\\JZX\\MPtrj_2022.9_full.json'
struJData = StructureJsonData(mptrj_path, r_cut=3.0, k=4)
dataloader = DataLoader(struJData, batch_size=1, shuffle=True)

# 下面这段缓存没测试，效率应该低爆了
# for i, data in tqdm(enumerate(dataloader)):
#     cache_path = os.path.join(cache_dir, f"data_{i}.pkl")
#     with open(cache_path, "wb") as f:
#         pickle.dump(data, f)


## Load data from cache
# cached_data = []
# for file_name in sorted(os.listdir(cache_dir)):
#     if file_name.endswith(".pkl"):
#         file_path = os.path.join(cache_dir, file_name)
#         with open(file_path, "rb") as f:
#             data = pickle.load(f)
#             cached_data.append(data)
# cached_dataloader = DataLoader(cached_data, batch_size=4, shuffle=True)
# for batch in cached_dataloader:
#     print(batch)
#     break
