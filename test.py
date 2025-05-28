import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import vgg19, VGG19_Weights

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

content_tensor = load_image(r'C:\Users\ayush\RnD_Cgs\content_image.jpg', max_size=512)
style_tensor = load_image(r'C:\Users\ayush\RnD_Cgs\style_image.jpeg', max_size=512)

print("After loading images:")
print("Content Image size:", content_tensor.size())  # e.g. torch.Size([1, 3, 512, 384])
print("Style Image size:", style_tensor.size())


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


content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

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
    return G.div(c * h * w)

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


def get_style_model_and_losses(cnn,
                               style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):

    content_losses = []
    style_losses = []
    style_img_preprocessed = preprocess_tensor(style_img)
    content_img_preprocessed = preprocess_tensor(content_img)
    
    print("After preprocessing:")
    print("Content Image size:", content_img_preprocessed.size())
    print("Style Image size:", style_img_preprocessed.size())
    # assuming that ``cnn`` is a ``nn.Sequential``, so we make a new ``nn.Sequential``
    # to put in modules that are supposed to be activated sequentially
    model = nn.Sequential() 

    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img_preprocessed).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img_preprocessed).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses

input_tensor = content_tensor.clone()

print("Input image size at start:", input_tensor.size())


plt.figure(figsize=(10, 5))
img_show(tensor_to_img(input_tensor), 'Input Image')
plt.show(block = False)

def get_input_optimizer(input_img):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img])
    return optimizer

def run_style_transfer(cnn, content_img, style_img, input_img, num_steps=300,
                       style_weight=1000000, content_weight=1):
    """Run the style transfer."""
    print('Building the style transfer model..')
    
    model, style_losses, content_losses = get_style_model_and_losses(cnn,style_img, content_img)

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_img.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.
    model.eval()
    model.requires_grad_(False)

    print("Input Image size:", input_tensor.size())
    
    optimizer = get_input_optimizer(input_img)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_img.clamp_(0, 1)

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

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img

num_steps = 100

print("After loading images:")
print("Content Image size:", content_tensor.size())  # e.g. torch.Size([1, 3, 512, 384])
print("Style Image size:", style_tensor.size())
print("Input Image size:", input_tensor.size())

output = run_style_transfer(cnn,content_tensor, style_tensor, input_tensor, num_steps)

plt.figure(figsize=(10, 5))
img_show(tensor_to_img(output), 'Output Image')    
plt.show()
plt.imsave(r'C:\Users\ayush\RnD_Cgs\output_image.jpg', tensor_to_img(output))