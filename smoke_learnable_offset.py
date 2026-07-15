import torch

from builder import data_builder, loss_builder, model_builder
from config.config import load_config_data


config = load_config_data("config/semanticposs_learnable_offset_b1_full.yaml")
model_config = config["model_params"]
dataset_config = config["dataset_params"]
device = torch.device("cuda:0")

model = model_builder.build(model_config).to(device)
train_loader, _ = data_builder.build(
    dataset_config,
    config["train_data_loader"],
    config["val_data_loader"],
    grid_size=model_config["output_shape"],
)
batch = next(iter(train_loader))
_, voxel_labels, grid_indices, _, point_features = batch
point_features = [torch.from_numpy(x).float().to(device) for x in point_features]
grid_indices = [torch.from_numpy(x).to(device) for x in grid_indices]
voxel_labels = voxel_labels.long().to(device)

output = model(point_features, grid_indices, voxel_labels.shape[0])
cross_entropy, lovasz = loss_builder.build(
    wce=True,
    lovasz=True,
    num_class=model_config["num_class"],
    ignore_label=dataset_config["ignore_label"],
)
loss = cross_entropy(output, voxel_labels) + lovasz(
    torch.nn.functional.softmax(output, dim=1), voxel_labels, ignore=0
)
loss.backward()

generator = model.cylinder_3d_generator
with torch.no_grad():
    raw_features = point_features[0]
    normalized_offset = torch.tanh(generator.offset_mlp(raw_features))

print("output_shape:", tuple(output.shape))
print("loss:", float(loss))
print("initial_offset_abs_max:", float(normalized_offset.abs().max()))
print("offset_head_grad_abs_sum:", float(generator.offset_mlp[-1].weight.grad.abs().sum()))
print("total_parameters:", sum(p.numel() for p in model.parameters()))
print("offset_parameters:", sum(p.numel() for p in generator.offset_mlp.parameters()))
print("peak_gpu_memory_gib:", torch.cuda.max_memory_allocated() / 1024 ** 3)
