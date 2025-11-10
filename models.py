import torch
import torch.nn as nn

class MicroChad(nn.Module):
    def __init__(self, out_count=2):
        super(MicroChad, self).__init__()
        self.conv1 = nn.Conv2d(4, 14, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(14, 21, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(21, 32, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(32, 47, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(47, 70, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(70, 106, kernel_size=3, stride=1, padding=1)
        self.fc_gaze = nn.Linear(106, out_count)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        self.adaptive = nn.AdaptiveMaxPool2d(output_size=1)

        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_blends=True):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv3(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv4(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv5(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv6(x)
        x = self.act(x)

        x = self.adaptive(x)
        x = torch.flatten(x, 1)
        
        x = self.fc_gaze(x)
        x = self.sigmoid(x)

        return x

class MultiInputMergedMicroChad(nn.Module):
    def __init__(self, left_models, right_models):
        super(MultiInputMergedMicroChad, self).__init__()

        if not left_models or not right_models:
            raise ValueError("Model lists for both left and right inputs cannot be empty.")

        self.all_models = left_models + right_models
        self.num_models = len(self.all_models)
        self.num_left = len(left_models)
        self.num_right = len(right_models)
        self.original_model_ref = self.all_models[0]

        # --- Create merged convolutional layers with input routing logic ---
        self._create_merged_conv_layers(left_models, right_models)

        # --- Store the final classification layers separately ---
        self.final_layers = nn.ModuleList(
            [model.fc_gaze for model in self.all_models]
        )

        # --- Define shared, parameter-less layers ---
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.adaptive = nn.AdaptiveMaxPool2d(output_size=1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def _create_merged_conv_layers(self, left_models, right_models):
        """Merges conv layers, with special handling for conv1 to route inputs."""
        
        conv_layer_names = [name for name, module in self.original_model_ref.named_modules() if isinstance(module, nn.Conv2d)]

        for i, layer_name in enumerate(conv_layer_names):
            # Get layers from all models for merging
            all_original_layers = [getattr(model, layer_name) for model in self.all_models]
            out_c, in_c, k_h, k_w = all_original_layers[0].weight.shape
            
            if i == 0:
                # --- SPECIAL HANDLING FOR THE FIRST CONV LAYER ---
                # This layer will perform the input routing.
                groups = 1 # It's a single, custom-built convolution
                
                # The merged layer will take all input channels (e.g., 4+4=8)
                new_in_c = in_c * 2 # Assuming two input streams (left, right)
                # The output channels are the sum of all model outputs
                new_out_c = out_c * self.num_models

                # Build the block-diagonal weight matrix for input routing
                merged_weight = torch.zeros(new_out_c, new_in_c, k_h, k_w)
                
                # Get weights for left and right models separately
                left_conv1_layers = [model.conv1 for model in left_models]
                right_conv1_layers = [model.conv1 for model in right_models]

                # Concatenate weights for each stream
                left_weights = torch.cat([layer.weight.data for layer in left_conv1_layers], dim=0)
                right_weights = torch.cat([layer.weight.data for layer in right_conv1_layers], dim=0)
                
                # Place left weights in the top-left block of the matrix
                # It will process the first 4 input channels (0:in_c)
                merged_weight[0:self.num_left*out_c, 0:in_c, :, :] = left_weights

                # Place right weights in the bottom-right block of the matrix
                # It will process the last 4 input channels (in_c:)
                merged_weight[self.num_left*out_c:, in_c:, :, :] = right_weights

                # Biases can just be concatenated as they are applied after the routing
                merged_bias = torch.cat([layer.bias.data for layer in all_original_layers], dim=0)

            else:
                # --- SUBSEQUENT CONV LAYERS ---
                # Use grouped convolutions for efficiency and separation.
                groups = self.num_models
                new_in_c = in_c * self.num_models
                new_out_c = out_c * self.num_models
                
                merged_weight = torch.cat([layer.weight.data for layer in all_original_layers], dim=0)
                merged_bias = torch.cat([layer.bias.data for layer in all_original_layers], dim=0)

            # Create the new, wider Conv2D layer
            merged_layer = nn.Conv2d(new_in_c, new_out_c, (k_h, k_w),
                                    stride=all_original_layers[0].stride,
                                    padding=all_original_layers[0].padding,
                                    groups=groups)
            
            merged_layer.weight.data = merged_weight
            merged_layer.bias.data = merged_bias
            setattr(self, layer_name, merged_layer)

    def forward(self, x):
        # The forward pass is simple because conv1 handles the routing.
        x = self.act(self.conv1(x))
        x = self.pool(x)

        x = self.act(self.conv2(x))
        x = self.pool(x)

        x = self.act(self.conv3(x))
        x = self.pool(x)
        
        x = self.act(self.conv4(x))
        x = self.pool(x)

        x = self.act(self.conv5(x))
        x = self.pool(x)

        x = self.act(self.conv6(x))
        
        x = self.adaptive(x)
        x = torch.flatten(x, 1)

        # Split the final feature vector and pass to respective heads
        outputs = []
        chunk_size = self.original_model_ref.conv6.out_channels
        feature_chunks = torch.split(x, split_size_or_sections=chunk_size, dim=1)
        
        for i, head in enumerate(self.final_layers):
            output = head(feature_chunks[i])
            output = self.sigmoid(output)
            outputs.append(output)
            
        return outputs
