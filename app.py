import gradio as gr
import torch
import torch.nn.functional as F
from robustbench.utils import load_model
from torchvision import transforms

# setup and configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {DEVICE}...")

# load models from RobustBench
print("Downloading/Loading Standard Model...")
standard_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(DEVICE)
standard_model.eval()

print("Downloading/Loading Robust Model...")
robust_model = load_model(model_name='Carmon2019Unlabeled', dataset='cifar10', threat_model='Linf').to(DEVICE)
robust_model.eval()

# CIFAR-10 class labels
LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# helper functions
def preprocess(image):
    """Converts Gradio image (numpy) to Torch Tensor [0,1] for RobustBench."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)), # resize to CIFAR-10 size
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0).to(DEVICE)

def fgsm_attack(model, image_tensor, epsilon):
    """Performs Fast Gradient Sign Method (FGSM) attack."""
    if epsilon == 0:
        return image_tensor
    
    image_tensor.requires_grad = True
    
    # forward pass
    output = model(image_tensor)
    init_pred = output.max(1, keepdim=True)[1] # get index of max log-probability
    
    # calculate loss (cross-entropy)
    loss = F.cross_entropy(output, init_pred[0])
    
    # zero gradients
    model.zero_grad()
    
    # backward pass
    loss.backward()
    
    # collect data gradient
    data_grad = image_tensor.grad.data
    
    # create the perturbed image
    sign_data_grad = data_grad.sign()
    perturbed_image = image_tensor + epsilon * sign_data_grad
    
    # clamp to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    
    return perturbed_image

def get_prediction(model, tensor):
    """Returns the top label and confidence dictionary."""
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
    
    # create dict for Gradio label component
    confidences = {LABELS[i]: float(probs[i]) for i in range(10)}
    return confidences

def amplify_noise(clean_tensor, adv_tensor):
    """Visualizes the noise by amplifying the difference."""
    noise = (adv_tensor - clean_tensor).abs()
    # scale up noise visibility 
    noise = noise * 50 
    noise = torch.clamp(noise, 0, 1)
    return transforms.ToPILImage()(noise.squeeze().cpu())

# main demo function
def run_demo(image, epsilon):
    if image is None:
        return None, None, None, None, None

    # prepare data
    clean_tensor = preprocess(image)
    
    # generate attack
    adv_tensor = fgsm_attack(standard_model, clean_tensor, epsilon)
    
    # get predictions
    # standard model on attacked image
    std_pred = get_prediction(standard_model, adv_tensor)
    
    # robust model on attacked image
    rob_pred = get_prediction(robust_model, adv_tensor)
    
    # visuals
    perturbed_img_display = transforms.ToPILImage()(adv_tensor.squeeze().cpu())
    noise_img_display = amplify_noise(clean_tensor, adv_tensor)
    
    return perturbed_img_display, noise_img_display, std_pred, rob_pred

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Adversarial Training Demo (RobustBench)")
    gr.Markdown("Compare a **Standard ResNet** vs. a **Robust ResNet** (Carmon2019) under attack.")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(label="Upload Image (Will be resized to 32x32)", type="pil")
            epsilon_slider = gr.Slider(0, 0.1, value=0.00, step=0.005, label="Attack Strength (Epsilon)")
            run_btn = gr.Button("Run Attack", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Visualization")
            # show the attacked image
            adv_image_out = gr.Image(label="Attacked Image")
            # show the noise pattern
            noise_image_out = gr.Image(label="The Invisible Noise (Amplified 50x)")
            
    gr.Markdown("### Model Predictions on Attacked Image")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Standard Model Prediction")
            std_label = gr.Label(num_top_classes=3)
        
        with gr.Column():
            gr.Markdown("### Robust Model Prediction")
            rob_label = gr.Label(num_top_classes=3)

    run_btn.click(
        fn=run_demo,
        inputs=[input_img, epsilon_slider],
        outputs=[adv_image_out, noise_image_out, std_label, rob_label]
    )

if __name__ == "__main__":
    demo.launch(debug=True, server_name="localhost")
