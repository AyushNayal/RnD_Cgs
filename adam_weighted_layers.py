import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)


def load_image(image_path, max_size=512, device='cuda' if torch.cuda.is_available() else 'cpu'):
    image = Image.open(image_path).convert('RGB')

    # Resize only if larger than max_size
    if max(image.size) > max_size:
        scale = max_size / max(image.size)
        new_size = tuple([int(dim * scale) for dim in image.size])
    else:
        new_size = image.size

    transform = transforms.Compose([
        transforms.Resize(new_size[::-1]),  # PIL size = (width, height), PyTorch expects (height, width)
        transforms.ToTensor(),
    ])

    image = transform(image).unsqueeze(0)
    return image.to(device)

content_tensor = load_image(r'C:\Users\ayush\RnD_Cgs\content_6.jpg', max_size=512)
style_tensor = load_image(r'C:\Users\ayush\RnD_Cgs\style_.jpg', max_size=512)


def tensor_to_img(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def preprocess_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

def undo_normalization(tensor, device=None):
    """
    Reverses VGG19 normalization on an image tensor.
    Assumes the input tensor is of shape (1, 3, H, W) or (3, H, W).
    
    Args:
        tensor (torch.Tensor): Normalized image tensor.
        device (str or torch.device): Optional device override.
    
    Returns:
        torch.Tensor: Unnormalized image tensor.
    """
    if device is None:
        device = tensor.device

    # Ensure tensor has shape (1, 3, H, W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)

    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    unnormalized = tensor * std + mean
    return unnormalized.clamp(0, 1)

def img_show(image,title = None):

    plt.imshow(image)
    if title is not None:
        plt.title(title)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
img_show(tensor_to_img(content_tensor), 'Content Image')    
plt.subplot(1, 2, 2)
img_show(tensor_to_img(style_tensor), 'Style Image')

plt.tight_layout()
plt.show(block =False)


content_layers = ['conv_4']
style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
intermediate_outputs = []

class ContentLoss(nn.Module):

    def __init__(self, target,):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        if input.shape != self.target.shape:
            raise ValueError(f"Shape mismatch: input {input.shape}, target {self.target.shape}")
        self.loss = F.mse_loss(input, self.target)
        return input

    
def gram_matrix(input):
    b, c, h, w = input.size()
    features = input.view(c, h * w)  # Flatten spatial dims
    G = torch.mm(features, features.t())
    return G.div(c * h * w )

class StyleLoss(nn.Module):
    def __init__(self, target_feature, weight=1.0):
        super(StyleLoss, self).__init__()
        self.target = gram_matrix(target_feature).detach()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = self.weight * nn.functional.mse_loss(G, self.target)
        return input
    
class TotalVariationLoss(nn.Module):
    def __init__(self, weight=1.0):
        super(TotalVariationLoss, self).__init__()
        self.weight = weight
        self.loss = 0

    def forward(self, input):
        self.loss = self.weight * (
            torch.sum(torch.abs(input[:, :, :, :-1] - input[:, :, :, 1:])) + 
            torch.sum(torch.abs(input[:, :, :-1, :] - input[:, :, 1:, :]))
        )
        return input

   
cnn = vgg19(weights=VGG19_Weights.DEFAULT).features.eval()

def get_features_from_sequential(model, img, layers_to_extract):
    features = {}
    x = img
    for name, layer in model._modules.items():
        x = layer(x)
        if name in layers_to_extract:
            features[name] = x.detach()
    return features



def get_style_model_and_losses(cnn, style_img, content_img,style_weights,tv_weight):
    cnn = cnn.to(device).eval()

    # Build base sequential model
    model = nn.Sequential()
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            raise RuntimeError(f'Unrecognized layer: {layer.__class__.__name__}')

        model.add_module(name, layer)

    # Extract feature maps to use as targets
    content_targets = get_features_from_sequential(model, content_img, content_layers)
    style_targets = get_features_from_sequential(model, style_img, style_layers)

    # Now rebuild with loss layers
    final_model = nn.Sequential()
    content_losses = []
    style_losses = []

    i = 0
    for name, layer in list(model._modules.items()):
        final_model.add_module(name, layer)

        if name in content_layers:
            target = content_targets[name]
            content_loss = ContentLoss(target)
            final_model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = style_targets[name]
            weight = style_weights.get(name)  # Default to 1.0 if not found
            style_loss = StyleLoss(target, weight=weight)
            final_model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        i += 1
    tv_loss_module = TotalVariationLoss(weight=tv_weight)
    final_model.add_module("tv_loss", tv_loss_module)
    # Trim model after the last loss layer
    # for i in range(len(final_model) - 1, -1, -1):
    #     if isinstance(final_model[i], ContentLoss) or isinstance(final_model[i], StyleLoss):
    #         break
    # final_model = final_model[:i + 1]

    return final_model, style_losses, content_losses, tv_loss_module

input_tensor = content_tensor.clone()


# plt.figure(figsize=(10, 5))
# img_show(tensor_to_img(input_tensor), 'Input Image')
# plt.show(block = False)

def get_input_optimizer(input_img, lr=0.02):
    optimizer = optim.Adam([input_img], lr=lr)
    return optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, style_weights, num_steps,
                       style_weight, content_weight, save_every,tv_weight, intermediate_outputs=[]):
    """Run the style transfer using Adam optimizer."""
    print('Building the style transfer model...')
    model, style_losses, content_losses,tv_loss_module = get_style_model_and_losses(cnn, style_img, content_img, style_weights,tv_weight)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = get_input_optimizer(input_img, lr=0.005)

    print('Optimizing with Adam...')
    

    for step in range(1, num_steps + 1):
        
        optimizer.zero_grad()

        # Preprocess input image
        # input_img_preprocessed = preprocess_tensor(input_img)
        model(input_img)

        style_score = sum(sl.loss for sl in style_losses)
        content_score = sum(cl.loss for cl in content_losses)
        
        style_score *= style_weight
        content_score *= content_weight

        tv_loss_module(input_img)
        tv_score = tv_loss_module.loss

        total_loss = style_score + content_score + tv_score
        total_loss.backward()
        optimizer.step()

        with torch.no_grad():
            input_img.clamp_(0, 1)

        if step % 50 == 0:
            print(f"Step {step}/{num_steps} - Total Loss : {total_loss:.4f} Style Loss: {style_score:.4f}, Content Loss: {content_score:.4f}")


        if step % save_every == 0:
            with torch.no_grad():
                intermediate_outputs.append((step, tensor_to_img(input_img.cpu().clone())))

    return input_img


num_steps = 5000*2
# style_weight = 1e7
# content_weight =1e1
# tv_weight = 30000
style_weights = {
    'conv_1': 0.1,
    'conv_2': 1.5,
    'conv_3': 1.5,
    'conv_4': 3.7,
    'conv_5': 4
    
}
# style_weights = {
#     'conv_1': 0.2,
#     'conv_2': 0.2,
#     'conv_3': 0.2,
#     'conv_4': 0.2,
#     'conv_5': 0.2
    
# }

style_weight = 1e6
content_weight = 2
tv_weight = 1e-5
# style_weights = {
#     'conv_1': 0.2,
#     'conv_2': 0.8,
#     'conv_3': 1.0,
#     'conv_4': 1.5,
#     'conv_5': 2.0
# }

intermediate_outputs = []  # to store (step, image) tuples
save_every = 1000*2

# input_tensor = preprocess_tensor(input_tensor)

output = run_style_transfer(cnn,content_tensor, style_tensor, input_tensor, style_weights, num_steps, style_weight, content_weight,save_every,tv_weight,intermediate_outputs)


num_images = len(intermediate_outputs)
fig, axs = plt.subplots(1, num_images, figsize=(5 * num_images, 5))

if num_images == 1:
    axs = [axs]  # ensure iterable if only one

for ax, (step, img) in zip(axs, intermediate_outputs):
    ax.imshow(img)
    ax.set_title(f'Step {step}')
    ax.axis('off')

plt.tight_layout()
plt.show(block = False)
# output = undo_normalization(output)
plt.figure(figsize=(12, 6))
img_show(tensor_to_img(output), 'Output Image')    
plt.show()
