import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

import copy


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

content_tensor = load_image(r'C:\Users\ayush\RnD_Cgs\WhatsApp Image 2025-05-18 at 11.43.42_ca882cd4.jpg', max_size=512)
style_tensor = load_image(r'C:\Users\ayush\RnD_Cgs\imggg.jpeg', max_size=512)


def tensor_to_img(tensor):
    image = tensor.cpu().clone()
    image = image.squeeze(0)
    image = transforms.ToPILImage()(image)
    return image


def preprocess_tensor(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(1, 3, 1, 1)
    return (tensor - mean) / std

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
    return G.div(c * h * w * 0.1)

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super(StyleLoss, self).__init__()
        # Compute and store Gram matrix of target style features
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)  # compute Gram matrix of input features
        self.loss = F.mse_loss(G, self.target)  # compare Gram matrices
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



def get_style_model_and_losses(cnn, style_img, content_img):
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
    for name, layer in model._modules.items():
        final_model.add_module(name, layer)

        if name in content_layers:
            target = content_targets[name]
            content_loss = ContentLoss(target)
            final_model.add_module(f"content_loss_{i}", content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target = style_targets[name]
            style_loss = StyleLoss(target)
            final_model.add_module(f"style_loss_{i}", style_loss)
            style_losses.append(style_loss)

        i += 1

    # Trim model after the last loss layer
    for i in range(len(final_model) - 1, -1, -1):
        if isinstance(final_model[i], ContentLoss) or isinstance(final_model[i], StyleLoss):
            break
    final_model = final_model[:i + 1]

    return final_model, style_losses, content_losses

input_tensor = content_tensor.clone()


# plt.figure(figsize=(10, 5))
# img_show(tensor_to_img(input_tensor), 'Input Image')
# plt.show(block = False)

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=5000,
                       style_weight=1e6, content_weight=2.0, lr=0.01):
    """Run the style transfer using Adam optimizer."""
    print('Building the style transfer model...')

    model, style_losses, content_losses = get_style_model_and_losses(cnn, style_img, content_img)

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    # Use Adam optimizer
    optimizer = optim.Adam([input_img], lr=lr)

    print('Optimizing using Adam...')

    for step in range(1, num_steps + 1):
        optimizer.zero_grad()
        input_img_preprocessed = preprocess_tensor(input_img)
        model(input_img_preprocessed)

        style_score = 0
        content_score = 0

        for sl in style_losses:
            style_score += sl.loss
        for cl in content_losses:
            content_score += cl.loss

        style_score *= style_weight
        content_score *= content_weight

        loss = style_score + content_score
        loss.backward()
        optimizer.step()

        # Clamp to [0, 1] after each step to avoid overflow
        with torch.no_grad():
            input_img.clamp_(0, 1)

        # Log every 100 iterations
        if step % 100 == 0 or step == num_steps:
            print(f"Step {step}/{num_steps} - Style Loss: {style_score.item():.4f}, Content Loss: {content_score.item():.4f}")

    return input_img


num_steps = 1000
style_weight = 1e6
content_weight = 2e1
lr = 0.02
output = run_style_transfer(cnn,content_tensor, style_tensor, input_tensor, num_steps, style_weight, content_weight,lr)

plt.figure(figsize=(10, 5))
img_show(tensor_to_img(output), 'Output Image')    
plt.show()
plt.imsave(r'C:\Users\ayush\RnD_Cgs\output_image.jpg', tensor_to_img(output))