from torch.utils.data import Dataset, DataLoader


class NotesDataset(Dataset): 
    
    def __init__(self, in_sequences, out_sequences):
        self.in_sequences = torch.from_numpy(in_sequences).float()
        self.in_sequences.to(device)
        self.out_sequences = torch.from_numpy(out_sequences)
        self.out_sequences.to(device)

    def __len__(self):
        return len(self.in_sequences)

    def __getitem__(self, idx): 
        return self.in_sequences[idx], self.out_sequences[idx]