import torch
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, LayerNorm, \
    BatchNorm, SAGEConv, GINEConv, GATv2Conv, JumpingKnowledge, TopKPooling, knn_interpolate
import torch.nn.functional as F
from torch_geometric.data import Data
import torch.nn as nn
import torch_geometric
from torch_scatter import scatter_add

"""
Model for Votex Net Graph Attention Network Model. 

Ref: P. Velickovic, https://arxiv.org/abs/1710.10903
Y. Shen, June 2024

Modification:
Y. Shen, August 10 2024: GNN1: model with GAT attention 
Y. Shen, September 20 2024: GNN2: model with edge encoding only
Y. Shen, September 30 2024: GNN3: model with attention and edge encoding


"""
class GNN(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, num_coarse=None, num_fine=None, dropout_rate=0.2):
        super(GNN, self).__init__()
        HEADS = 6
        
        self.node_encoder = torch.nn.Linear(node_in_channels, hidden_channels)
        self.edge_encoder1 = torch.nn.Linear(edge_in_channels, hidden_channels * hidden_channels)
        self.edge_encoder2 = torch.nn.Linear(edge_in_channels, hidden_channels * hidden_channels)
        self.edge_encoder3 = torch.nn.Linear(edge_in_channels, hidden_channels * out_channels)
        
        self.conv1 = GATConv(hidden_channels, hidden_channels, heads=HEADS, concat=True)
        self.conv2 = GATConv(hidden_channels * HEADS, hidden_channels, heads=HEADS, concat=True)
        self.conv3 = GATConv(hidden_channels * HEADS, out_channels, heads=1, concat=False)
        
        self.norm1 = BatchNorm(hidden_channels * HEADS)
        self.norm2 = BatchNorm(hidden_channels * HEADS)
        self.norm3 = BatchNorm(out_channels)
        
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        if num_coarse != num_fine:
            self.fc = torch.nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None
        self.dropout_rate = dropout_rate    

        self.fc1 = torch.nn.Linear(hidden_channels, out_channels)
        self.fc2 = torch.nn.Linear(hidden_channels * HEADS, hidden_channels * HEADS)
        self.fc3 = torch.nn.Linear(hidden_channels * HEADS, out_channels)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        residual0 = x[:, 0:1]
        
        x = self.node_encoder(x)
        residual1 = self.fc1(x)
        
        x = self.conv1(x, edge_index)
        x = self.norm1(x)
        x = F.relu(x)
        #x = F.dropout(x, p=self.dropout_rate, training=self.training)
        #residual2 = self.fc2(x)
        
        x = self.conv2(x, edge_index)
        x = self.norm2(x)
        x = F.relu(x)
        #x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # add skip connection of residual2
        #x += residual2
        x = self.conv3(x, edge_index)
        x = self.norm3(x)
        #x = F.tanh(x)
        
        # Add the residual connection
        x += residual1    
        #x += residual2
        #x += residual3

        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)

        # Add the residual connection
        x += residual0    

        
        
        return x



class GNN2(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, 
                 num_coarse=None, num_fine=None, dropout_rate=0.2):
        super(GNN2, self).__init__()
        
        # Node and Edge Encoders
        self.node_encoder = torch.nn.Linear(node_in_channels, hidden_channels)
        self.edge_encoder = torch.nn.Linear(edge_in_channels, hidden_channels)
        
        # Define the MLPs for GINEConv layers
        nn1 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        nn2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_channels, hidden_channels)
        )
        nn3 = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels)
        )
        
        # Convolutional Layers using GINEConv
        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)
        self.conv3 = GINEConv(nn3)
        
        # Batch Normalization Layers
        self.norm1 = BatchNorm(hidden_channels)
        self.norm2 = BatchNorm(hidden_channels)
        self.norm3 = BatchNorm(out_channels)
        
        # Optional Fully Connected Layer
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        if num_coarse != num_fine:
            self.fc = torch.nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None
        
        self.dropout_rate = dropout_rate    
        
        # Linear Layer for Residual Connection
        self.fc1 = torch.nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        # Extract node features, edge indices, and edge attributes
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        # Capture the first feature for residual connection
        residual0 = x[:, 0:1]
        
        # Encode node features
        x = self.node_encoder(x)
        
        # Encode edge features
        edge_attr = self.edge_encoder(edge_attr)
        
        # Capture residual after node encoding
        residual1 = self.fc1(x)
        
        # First GINEConv Layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second GINEConv Layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Third GINEConv Layer
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        
        # Add residual connection
        x += residual1
        
        # Optional fully connected layer
        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)
        
        # Add the initial feature as residual
        x += residual0
        
        return x


class GNN3(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels, 
                 num_coarse=None, num_fine=None, dropout_rate=0.2, HEADS=6, ALPHA0 = 0.5, ALPHA1 = 0.5):

        super().__init__()
        
        # Node and Edge Encoders
        self.node_encoder = torch.nn.Linear(node_in_channels, hidden_channels * hidden_channels)
        self.edge_encoder = torch.nn.Linear(edge_in_channels, hidden_channels)
        
        # GATv2Conv Layers with edge_dim
        self.conv1 = GATv2Conv(hidden_channels * hidden_channels, hidden_channels, edge_dim=hidden_channels, heads=HEADS, concat=True)
        self.conv2 = GATv2Conv(hidden_channels * HEADS, hidden_channels, edge_dim=hidden_channels, heads=HEADS, concat=True)
        self.conv3 = GATv2Conv(hidden_channels * HEADS, out_channels, edge_dim=hidden_channels, heads=1, concat=False)
        
        # Batch Normalization Layers
        self.norm1 = BatchNorm(hidden_channels  * HEADS)
        self.norm2 = BatchNorm(hidden_channels  * HEADS)
        self.norm3 = BatchNorm(out_channels)
        
        # Optional Fully Connected Layer
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        if num_coarse != num_fine:
            self.fc = torch.nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None
        
        self.dropout_rate = dropout_rate    
        
        # Linear Layer for Residual Connection
        self.fc1 = torch.nn.Linear(hidden_channels * hidden_channels, out_channels)
        self.fc2 = torch.nn.Linear(hidden_channels * HEADS, hidden_channels * HEADS)
        self.fc3 = torch.nn.Linear(hidden_channels * HEADS, out_channels)

        # Residual connection weights 
        # Add these lines in your __init__ method
        # Add these lines in your __init__ method
        self.alpha0_net = torch.nn.Sequential(
            torch.nn.Linear(node_in_channels, 1),
            torch.nn.Sigmoid()  # Ensures the output is between 0 and 1
        )
        self.alpha1_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels * hidden_channels, 1),
            torch.nn.Sigmoid()
        )

        self.ALPHA0 = ALPHA0
        self.ALPHA1 = ALPHA1

        
    def forward(self, data):
        # train small MLP on weights for skip connection 
        alpha0 = self.alpha0_net(data.x) * self.ALPHA0 # this is the first skip connection weight 
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        
        residual0 = x[:, 0:1] # this is the VLM DCp
        
        # Encode node and edge features
        x = self.node_encoder(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        edge_attr = self.edge_encoder(edge_attr)
        
        # Residual Connection
        residual1 = self.fc1(x)
        alpha1 = self.alpha1_net(x) * self.ALPHA1 # this is the second skip connection weight
        
        # First GATv2Conv Layer
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Second GATv2Conv Layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Third GATv2Conv Layer
        x = self.conv3(x, edge_index, edge_attr)
        x = self.norm3(x)
        
        # Add residual connection
        x = alpha1 * residual1 + (1 - alpha1) * x
        
        # Optional fully connected layer
        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)
        
        # Add the residual connection
        x += residual0    

        return x


class GNN4(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels,
                 num_coarse=None, num_fine=None, dropout_rate=0.2, HEADS=6, ALPHA=0.5, HOP=3):
        super().__init__()

        self.HOP = HOP
        self.ALPHA = ALPHA

        # Node and Edge Encoders
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels * hidden_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)

        # Initialize lists to store layers and output dimensions
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.layer_output_dims = []

        # Create GNN layers
        for i in range(HOP):
            if i == 0:
                # First layer input dimension
                in_channels = hidden_channels * hidden_channels  # Output of node_encoder
            else:
                in_channels = hidden_channels * HEADS  # Output of previous layer

            if i == HOP - 1:
                # Last layer output dimension
                out_channels_layer = out_channels
                heads = 1
                concat = False
            else:
                out_channels_layer = hidden_channels
                heads = HEADS
                concat = True

            # Define GATv2Conv layer
            conv = GATv2Conv(in_channels, out_channels_layer,
                             edge_dim=hidden_channels, heads=heads, concat=concat)
            self.conv_layers.append(conv)

            # Define BatchNorm layer
            if concat:
                norm_layer = BatchNorm(out_channels_layer * heads)
                layer_dim = out_channels_layer * heads
            else:
                norm_layer = BatchNorm(out_channels_layer)
                layer_dim = out_channels_layer
            self.norm_layers.append(norm_layer)
            self.layer_output_dims.append(layer_dim)

        # Define U-Net-style skip connections and linear layers for dimension adjustment
        self.skip_connections = {}
        self.skip_linear_layers = nn.ModuleDict()

        for i in range(HOP // 2):
            encoder_layer = i
            decoder_layer = HOP - i - 1
            self.skip_connections[encoder_layer] = decoder_layer

            # Get output dimensions
            encoder_dim = self.layer_output_dims[encoder_layer]
            decoder_dim = self.layer_output_dims[decoder_layer]

            if encoder_dim != decoder_dim:
                # Create a linear layer to map from encoder_dim to decoder_dim
                self.skip_linear_layers[str(encoder_layer)] = nn.Linear(encoder_dim, decoder_dim)
            else:
                # No need for linear layer if dimensions match
                self.skip_linear_layers[str(encoder_layer)] = None

        # Optional Fully Connected Layer
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        if num_coarse != num_fine:
            self.fc = nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None

        self.dropout_rate = dropout_rate

    def forward(self, data, return_latent_space = False, 
                modify_latent=False, layer_to_modify=None, 
                modified_latent=None):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode node and edge features
        x = self.node_encoder(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        edge_attr = self.edge_encoder(edge_attr)

        layer_outputs = []
        encoder_outputs = {}  # To store outputs for skip connections
        latent_space_outputs = []
        # Forward through GNN layers
        for i, conv in enumerate(self.conv_layers):
            x = conv(x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

            # Store x for skip connections if needed (after convolution and activation)
            if i in self.skip_connections:
                encoder_outputs[i] = x


            if return_latent_space:
                latent_space_outputs.append(x.clone())

            if modify_latent and i == layer_to_modify:
                # Replace x with the modified latent representation
                x = modified_latent    
                
            layer_outputs.append(x)    

        # Apply U-Net-style skip connections
        for encoder_layer, decoder_layer in self.skip_connections.items():
            x_encoder = encoder_outputs[encoder_layer]
            x_decoder = layer_outputs[decoder_layer]

            # Get the linear layer for dimension adjustment, if any
            linear_layer = self.skip_linear_layers[str(encoder_layer)]

            if linear_layer is not None:
                x_encoder = linear_layer(x_encoder)

            # Add to decoder layer output
            layer_outputs[decoder_layer] = (1-self.ALPHA)*x_decoder + self.ALPHA*x_encoder

        # Output is from the last layer
        x = layer_outputs[-1]

        # Optional fully connected layer
        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)

        if return_latent_space:
            return x, latent_space_outputs
        else:
            return x

class gPool(nn.Module):
    """Graph Pooling (gPool) layer from Graph U-Nets paper"""
    def __init__(self, in_channels, ratio=0.7):
        super(gPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.p = nn.Parameter(torch.zeros(1, in_channels))
        nn.init.xavier_uniform_(self.p)
        
    def forward(self, x, edge_index, edge_attr=None, batch=None):
        # Calculate projection scores
        score = torch.matmul(x, self.p.t())
        score = torch.tanh(score)
        
        # Select top k nodes
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        num_nodes = scatter_add(torch.ones_like(batch), batch, dim=0)
        batch_size, max_num_nodes = num_nodes.size(0), num_nodes.max().item()
        
        k = (num_nodes * self.ratio).ceil().to(torch.long)
        
        # Get indices of top k nodes
        indices = []
        start = 0
        for i in range(batch_size):
            num = num_nodes[i]
            scores_batch = score[start:start + num]
            _, idx = torch.topk(scores_batch, k[i])
            indices.append(idx + start)
            start += num
            
        idx = torch.cat(indices)
        
        # Select the top k nodes and their features
        x = x[idx]
        batch = batch[idx]
        
        # Store indices for unpooling
        self.idx_select = idx
        
        # Update edge indices and attributes
        row, col = edge_index
        mask = (row.view(-1, 1) == idx.view(1, -1)).any(dim=1)
        mask &= (col.view(-1, 1) == idx.view(1, -1)).any(dim=1)
        
        edge_index = edge_index[:, mask]
        edge_attr = edge_attr[mask] if edge_attr is not None else None
        
        # Remap indices
        _, col = edge_index
        idx_remap = torch.zeros_like(score)
        idx_remap[idx] = torch.arange(idx.size(0), device=idx.device)
        edge_index = idx_remap[edge_index]
        
        return x, edge_index, edge_attr, batch, idx

class gUnpool(nn.Module):
    """Graph Unpooling (gUnpool) layer from Graph U-Nets paper"""
    def __init__(self):
        super(gUnpool, self).__init__()
    
    def forward(self, x, idx, size):
        """
        Args:
            x: Node features
            idx: Indices from corresponding gPool layer
            size: Size of original feature matrix
        """
        new_x = x.new_zeros(size)
        new_x[idx] = x
        return new_x

class GNN5(torch.nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels,
                 num_coarse=None, num_fine=None, dropout_rate=0.2, HEADS=6, ALPHA=0.5, HOP=3):
        super(GNN5, self).__init__()
        
        # Store parameters
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.dropout_rate = dropout_rate
        self.HEADS = HEADS
        self.ALPHA = ALPHA
        self.HOP = HOP
        
        # Calculate dimensions
        # Each head will process hidden_channels/HEADS features
        self.hidden_per_head = hidden_channels // HEADS
        # After concatenation, we'll have hidden_per_head * HEADS features
        self.hidden_concat = self.hidden_per_head * HEADS
        
        # Node and edge encoders
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)
        
        # Conv layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # First layer: hidden_channels -> hidden_per_head * HEADS
        self.conv_layers.append(
            GATv2Conv(hidden_channels, self.hidden_per_head, 
                     edge_dim=hidden_channels, heads=HEADS, concat=True)
        )
        self.norm_layers.append(BatchNorm(self.hidden_concat))
        
        # Middle layers: hidden_per_head * HEADS -> hidden_per_head * HEADS
        for i in range(1, HOP - 1):
            self.conv_layers.append(
                GATv2Conv(self.hidden_concat, self.hidden_per_head,
                         edge_dim=hidden_channels, heads=HEADS, concat=True)
            )
            self.norm_layers.append(BatchNorm(self.hidden_concat))
        
        # Last layer: hidden_per_head * HEADS -> out_channels
        self.conv_layers.append(
            GATv2Conv(self.hidden_concat, out_channels,
                     edge_dim=hidden_channels, heads=1, concat=False)
        )
        self.norm_layers.append(BatchNorm(out_channels))
        
        # Pooling layers
        self.pool_layers = nn.ModuleList()
        self.num_encoder_layers = HOP // 2
        
        # Add pooling layers for encoder part
        for i in range(self.num_encoder_layers):
            pool = TopKPooling(
                self.hidden_concat,  # Use concatenated hidden dimension
                ratio=0.7,
                multiplier=1.0,
                nonlinearity=torch.tanh
            )
            self.pool_layers.append(pool)
            
        # Optional fully connected layer
        if num_coarse != num_fine:
            self.fc = nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None
            
    def forward(self, data, return_latent_space=False, 
                modify_latent=False, layer_to_modify=None, 
                modified_latent=None):
        x, edge_index = data.x, data.edge_index
        
        # Handle edge attributes
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Handle batch information - ensure it's always a tensor
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode node features
        x = self.node_encoder(x)
        encoded_features = x.clone()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Encode edge features if they exist
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            # Create a default edge attribute tensor filled with zeros
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, self.hidden_channels, device=x.device)
        
        # Store features and connectivity for unpooling
        unpool_info = []
        # Store encoder outputs for skip connections
        encoder_features = []
        # Store pooling indices and scores
        pool_indices = []
        
        # Encoder Path
        for i in range(self.num_encoder_layers):
            # Apply convolution
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Store features for skip connection before pooling
            encoder_features.append(x.clone())
            
            # Store pre-pooling info
            unpool_info.append((
                x.clone(),
                edge_index.clone(),
                edge_attr.clone(),
                batch.clone()
            ))
            
            # Apply pooling
            x, edge_index, edge_attr, batch, perm, score = self.pool_layers[i](
                x, edge_index, edge_attr, batch
            )
            pool_indices.append((perm, score))
        
        # Middle layers (if any)
        for i in range(self.num_encoder_layers, self.HOP - self.num_encoder_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Decoder Path
        for i in range(self.HOP - self.num_encoder_layers, self.HOP):
            # Get corresponding encoder layer index
            encoder_idx = self.HOP - i - 1
            
            # Get original features and connectivity
            orig_x, orig_edge_index, orig_edge_attr, orig_batch = unpool_info[encoder_idx]
            
            # Get corresponding encoder features
            encoder_feat = encoder_features[encoder_idx]
            
            # Create new tensor with original size
            new_x = torch.zeros_like(orig_x)
            # Get pooling indices
            perm, score = pool_indices[encoder_idx]
            # Copy features to their original positions
            new_x[perm] = x
            # Fill remaining positions with mean of existing features
            if perm.size(0) > 0:
                mean_features = x.mean(dim=0)
                mask = torch.ones(new_x.size(0), dtype=torch.bool, device=x.device)
                mask[perm] = False
                new_x[mask] = mean_features
            x = new_x
            
            # Restore original connectivity and edge attributes
            edge_index = orig_edge_index
            edge_attr = orig_edge_attr
            batch = orig_batch
            
            # Apply convolution
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Add skip connection from corresponding encoder layer
            if i < self.HOP - 1:  # All but the last layer
                x = (1 - self.ALPHA) * x + self.ALPHA * encoder_feat
        
        # Final skip connection from input to output
        # Project encoded_features to match output dimension if needed
        if encoded_features.size(-1) != x.size(-1):
            encoded_features = F.linear(encoded_features, torch.randn(x.size(-1), encoded_features.size(-1), device=x.device))
        x = (1 - self.ALPHA) * x + self.ALPHA * encoded_features
        
        # Optional fully connected layer
        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)
        
        if return_latent_space:
            return x, encoder_features
        else:
            return x





class GNN6(nn.Module):
    def __init__(self, node_in_channels, edge_in_channels, hidden_channels, out_channels,
                 num_coarse=None, num_fine=None, dropout_rate=0.2, HEADS=6, ALPHA=0.5, HOP=3):
        super(GNN6, self).__init__()
        
        # Store parameters
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_coarse = num_coarse
        self.num_fine = num_fine
        self.dropout_rate = dropout_rate
        self.HEADS = HEADS
        self.ALPHA = ALPHA
        self.HOP = HOP

        # Now add a small learnable offset and gain:
        self.pool_multiplier = nn.Parameter(torch.tensor(float(1.0)))
        
        # Calculate dimensions
        # Each head will process hidden_channels/HEADS features
        self.hidden_per_head = hidden_channels // HEADS
        # After concatenation, we'll have hidden_per_head * HEADS features
        self.hidden_concat = self.hidden_per_head * HEADS
        
        # Node and edge encoders
        self.node_encoder = nn.Linear(node_in_channels, hidden_channels)
        self.edge_encoder = nn.Linear(edge_in_channels, hidden_channels)
        
        # Conv layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        # First layer: hidden_channels -> hidden_per_head * HEADS
        self.conv_layers.append(
            GATv2Conv(hidden_channels, self.hidden_per_head, 
                     edge_dim=hidden_channels, heads=HEADS, concat=True)
        )
        self.norm_layers.append(BatchNorm(self.hidden_concat))
        
        # Middle layers: hidden_per_head * HEADS -> hidden_per_head * HEADS
        for i in range(1, HOP - 1):
            self.conv_layers.append(
                GATv2Conv(self.hidden_concat, self.hidden_per_head,
                         edge_dim=hidden_channels, heads=HEADS, concat=True)
            )
            self.norm_layers.append(BatchNorm(self.hidden_concat))
        
        # Last layer: hidden_per_head * HEADS -> out_channels
        self.conv_layers.append(
            GATv2Conv(self.hidden_concat, out_channels,
                     edge_dim=hidden_channels, heads=1, concat=False)
        )
        self.norm_layers.append(BatchNorm(out_channels))
        
        # Pooling layers
        self.pool_layers = nn.ModuleList()
        self.num_encoder_layers = HOP // 2
        
        # Add pooling layers for encoder part
        for i in range(self.num_encoder_layers):
            pool = TopKPooling(
                self.hidden_concat,  # Use concatenated hidden dimension
                ratio=0.8,
                multiplier=self.pool_multiplier,
                nonlinearity=torch.tanh
            )
            self.pool_layers.append(pool)
            
        # Optional fully connected layer
        if num_coarse is not None and num_coarse != num_fine:
            self.fc = nn.Linear(num_coarse * num_coarse, num_fine * num_fine)
        else:
            self.fc = None
            
    def forward(self, data, return_latent_space=False, 
                modify_latent=False, layer_to_modify=None, 
                modified_latent=None):
        x, edge_index = data.x, data.edge_index
        
        # Handle edge attributes
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        
        # Handle batch information - ensure it's always a tensor
        if hasattr(data, 'batch') and data.batch is not None:
            batch = data.batch
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Encode node features
        x = self.node_encoder(x)
        encoded_features = x.clone()
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Encode edge features if they exist
        if edge_attr is not None:
            edge_attr = self.edge_encoder(edge_attr)
        else:
            # Create a default edge attribute tensor filled with zeros
            num_edges = edge_index.size(1)
            edge_attr = torch.zeros(num_edges, self.hidden_channels, device=x.device)
        
        # Store features and connectivity for unpooling
        unpool_info = []
        # Store encoder outputs for skip connections
        encoder_features = []
        # Store pooling indices and scores
        pool_indices = []
        
        # Encoder Path
        for i in range(self.num_encoder_layers):
            # Apply convolution
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Store features for skip connection before pooling
            encoder_features.append(x.clone())
            
            # Store pre-pooling info (for reversible unpool)
            unpool_info.append((
                x.clone(),
                edge_index.clone(),
                edge_attr.clone(),
                batch.clone()
            ))
            
            # Apply pooling
            x, edge_index, edge_attr, batch, perm, score = self.pool_layers[i](
                x, edge_index, edge_attr, batch
            )
            pool_indices.append((perm, score))
        
        # Middle layers (if any)
        for i in range(self.num_encoder_layers, self.HOP - self.num_encoder_layers):
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
        
        # Decoder Path
        for i in range(self.HOP - self.num_encoder_layers, self.HOP):
            # Get corresponding encoder layer index
            encoder_idx = self.HOP - i - 1
            
            # Get original features and connectivity
            orig_x, orig_edge_index, orig_edge_attr, orig_batch = unpool_info[encoder_idx]
            
            # Get corresponding encoder features
            encoder_feat = encoder_features[encoder_idx]
            
            # Create new tensor with original size
            new_x = torch.zeros_like(orig_x)
            # Get pooling indices
            perm, score = pool_indices[encoder_idx]
            
            # Copy pooled features back to their original positions
            new_x[perm] = x
            
            # Restore dropped node features exactly from orig_x
            mask = torch.ones(new_x.size(0), dtype=torch.bool, device=x.device)
            mask[perm] = False
            new_x[mask] = orig_x[mask]
            
            x = new_x
            
            # Restore original connectivity and edge attributes
            edge_index = orig_edge_index
            edge_attr = orig_edge_attr
            batch = orig_batch
            
            # Apply convolution
            x = self.conv_layers[i](x, edge_index, edge_attr)
            x = self.norm_layers[i](x)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)
            
            # Add skip connection from corresponding encoder layer
            if i < self.HOP - 1:  # All but the last layer
                x = (1 - self.ALPHA) * x + self.ALPHA * encoder_feat
        
        # Final skip connection from input to output
        # Project encoded_features to match output dimension if needed
        if encoded_features.size(-1) != x.size(-1):
            encoded_features = F.linear(encoded_features, torch.randn(x.size(-1), encoded_features.size(-1), device=x.device))
        x = (1 - self.ALPHA) * x + self.ALPHA * encoded_features
        
        # Optional fully connected layer
        if self.fc:
            x = x.view(1, -1)
            x = self.fc(x)
            x = x.view(-1, 1)


        
        if return_latent_space:
            return x, encoder_features
        else:
            return x
        