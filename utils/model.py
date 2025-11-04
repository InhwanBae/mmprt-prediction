import torch
import torch.nn as nn



class GroupedGraphAttentionNet(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(128, 32), encoder_dim=4, attention_method='GAT', attention_dim=8, multi_head=2, output_dim=2, feature_cols=[], group_cols={}, verbose=False):
        super().__init__()
        layers = []
        self.encoder_dim = encoder_dim
        self.attention_method = attention_method
        self.attention_dim = attention_dim
        self.multi_head = multi_head

        # Normalization parameters
        self.norm_mean = nn.Parameter(torch.zeros(input_dim))
        self.norm_std = nn.Parameter(torch.ones(input_dim))
        self.norm_samples = 0

        # Build MLP layers
        layers = nn.Sequential()
        prev_dim = input_dim
        for h in hidden_layer_sizes:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))  # add dropout for regularization
            prev_dim = h
        self.mlp_net = layers

        # Build grouped feature encoder
        self.group_index = []
        used_indices = set()
        for group_name, group_features in group_cols.items():
            group_indices = [feature_cols.index(feat) for feat in group_features if feat in feature_cols]
            if group_indices:
                self.group_index.append(group_indices)
                used_indices.update(group_indices)
        self.n_groups = len(self.group_index)

        remaining_indices = [i for i in range(len(feature_cols)) if i not in used_indices]
        if remaining_indices:
            print(f"INFO: Remaining features not in any group: {[feature_cols[i] for i in remaining_indices]}")

        grouped_encoder = []
        for group in self.group_index:
            layers = nn.Sequential()
            layers.append(nn.Linear(len(group), encoder_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            grouped_encoder.append(layers)
        self.grouped_encoder = nn.ModuleList(grouped_encoder)

        # Build graph attention edge
        if self.attention_method == 'GAT':
            self.attention_src = nn.Linear(encoder_dim, attention_dim * multi_head)
            self.attention_dst = nn.Linear(encoder_dim, attention_dim * multi_head)
        elif self.attention_method == 'Transformer':
            self.attention_query = nn.Linear(encoder_dim, attention_dim * multi_head)
            self.attention_key = nn.Linear(encoder_dim, attention_dim * multi_head)
            self.attention_value = nn.Linear(encoder_dim, attention_dim * multi_head)
        else:
            raise ValueError(f"Unknown attention method: {self.attention_method}")

        # Build classification head
        if self.attention_method == 'GAT':
            self.decoder = nn.Linear(hidden_layer_sizes[-1] + encoder_dim * self.n_groups + encoder_dim * self.n_groups, output_dim)
        elif self.attention_method == 'Transformer':
            self.decoder = nn.Linear(hidden_layer_sizes[-1] + encoder_dim * self.n_groups + attention_dim * self.n_groups, output_dim)

        # Print model summary
        if verbose:
            print("=== Model Summary ===")
            print('feature_cols:', feature_cols)
            print('group_cols:', group_cols)
            print(f'Number of groups: {self.n_groups}')
            print(f'Unused feature indices for groups: {remaining_indices}')
            print(f'Number of learnable parameters: {self.count_learnable_params()}')

    def forward(self, x):
        """
        x: Tensor of shape [batch_size, input_dim]
        returns: logits of shape [batch_size, output_dim]
        """
        x_hat = self.mlp_net(x)  # [batch_size, hidden_layer_sizes[-1]]

        x = self.normalize(x)
        # grouped encoder
        group_outputs = []
        for group, encoder in zip(self.group_index, self.grouped_encoder):
            group_output = encoder(x[:, group])
            group_outputs.append(group_output)
        group_outputs = torch.stack(group_outputs, dim=1)

        # Graph attention mechanism
        if self.attention_method == 'GAT':
            src = self.attention_src(group_outputs).view(-1, self.n_groups, self.multi_head, self.attention_dim).permute(0, 2, 1, 3)  # [batch_size, multi_head, n_groups, attention_dim]
            dst = self.attention_dst(group_outputs).view(-1, self.n_groups, self.multi_head, self.attention_dim).permute(0, 2, 1, 3)  # [batch_size, multi_head, n_groups, attention_dim]

            # attention_scores = torch.einsum('bhqd,bhkd->bhqk', src, dst)
            attention_scores = src @ dst.transpose(-2, -1)  # Faster than einsum
            attention_scores = attention_scores / (self.attention_dim ** 0.5)
            attention_weights = nn.functional.softmax(attention_scores, dim=-1)
            # attention_output = torch.einsum('bhqk,bkd->bhqd', attention_weights, group_outputs)
            attention_output = attention_weights @ group_outputs.unsqueeze(dim=1)  # Faster than einsum
            attention_output = attention_output.sum(dim=1)  # Sum over multi-head dimension

        elif self.attention_method == 'Transformer':
            query = self.attention_query(group_outputs).view(-1, self.n_groups, self.multi_head, self.attention_dim).permute(0, 2, 1, 3)  # [batch_size, multi_head, n_groups, attention_dim]
            key = self.attention_key(group_outputs).view(-1, self.n_groups, self.multi_head, self.attention_dim).permute(0, 2, 1, 3)  # [batch_size, multi_head, n_groups, attention_dim]
            value = self.attention_value(group_outputs).view(-1, self.n_groups, self.multi_head, self.attention_dim).permute(0, 2, 1, 3)  # [batch_size, multi_head, n_groups, attention_dim]

            # attention_scores = torch.einsum('bhqd,bhkd->bhqk', query, key)
            attention_scores = query @ key.transpose(-2, -1)  # Faster than einsum
            attention_scores = attention_scores / (self.attention_dim ** 0.5)
            attention_weights = nn.functional.softmax(attention_scores, dim=-1)
            # attention_output = torch.einsum('bhqk,bhkd->bhqd', attention_weights, value)
            attention_output = attention_weights @ value  # Faster than einsum
            attention_output = attention_output.sum(dim=1)  # Sum over multi-head dimension
        
        attention_output = attention_output.reshape(attention_output.size(0), -1)  # Flatten for concatenation

        # Classification head
        out = torch.cat([x_hat, group_outputs.view(-1, self.n_groups * self.encoder_dim), attention_output], dim=1)  # Residual connection
        out = self.decoder(out)
        return out
    
    def normalize(self, input_batch):
        input_batch = (input_batch - self.norm_mean.detach()) / (self.norm_std + 1e-8).detach()
        return input_batch.detach()
    
    def denormalize(self, output_batch):
        output_batch = output_batch * self.norm_std.detach() + self.norm_mean.detach()
        return output_batch
    
    def set_norm_mean_std(self, batch):
        self.norm_mean.data = batch.mean(dim=0)
        self.norm_std.data = batch.std(dim=0)
        self.norm_samples = batch.size(0)
    
    def update_norm_mean_std(self, batch):
        batch_mean = torch.mean(batch, dim=0)
        batch_std = torch.std(batch, dim=0)
        batch_size = batch.size(0)
        if torch.all(self.norm_mean.data == 0) and torch.all(self.norm_std.data == 1):
            # Initialize with the first batch statistics
            self.norm_mean.data = batch_mean
            self.norm_std.data = batch_std
        else:
            # Using running mean and std formula
            ratio = batch_size / (self.norm_samples + batch_size)
            self.norm_mean.data = (1 - ratio) * self.norm_mean.data + ratio * batch_mean
            self.norm_std.data = (1 - ratio) * self.norm_std.data + ratio * batch_std
            self.norm_samples = self.norm_samples + batch_size

    def count_learnable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
