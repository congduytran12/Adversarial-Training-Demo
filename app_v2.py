from gradio.components import image_editor
import gradio as gr
import torch
import torch.nn.functional as F
from robustbench.utils import load_model
from autoattack import AutoAttack
from torchvision import transforms
from PIL import Image

# setup and configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running on {DEVICE}...")

# load models from RobustBench
print("Loading Standard Model...")
standard_model = load_model(model_name='Standard', dataset='cifar10', threat_model='Linf').to(DEVICE)
standard_model.eval()

ROBUST_MODELS = {}

# CIFAR-10 class labels
LABELS = ['Airplane', 'Automobile', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

# model and attack choices
MODEL_CHOICES = {
    "Bartoldson2024 WRN-94-16": "Bartoldson2024Adversarial_WRN-94-16",
    "Amini2024 MeanSparse WRN-94-16": "Amini2024MeanSparse_S-WRN-94-16",
    "Bartoldson2024 WRN-82-8": "Bartoldson2024Adversarial_WRN-82-8"
}

ATTACK_CHOICES = {
    "FGSM (Fast)": "fgsm",
    "APGD-CE (Strong)": "apgd-ce",
    "APGD-DLR": "apgd-dlr",
    "FAB": "fab",
    "Square": "square",
    "APGD-T (Targeted)": "apgd-t",
    "FAB-T (Targeted)": "fab-t"
}

# helper functions
def load_robust_model(model_name):
    """Lazy load and cache robust models."""
    if model_name not in ROBUST_MODELS:
        print(f"Loading Robust Model: {model_name}...")
        model = load_model(model_name=model_name, dataset='cifar10', threat_model='Linf').to(DEVICE)
        model.eval()
        ROBUST_MODELS[model_name] = model
    return ROBUST_MODELS[model_name]

def preprocess(image):
    """Converts Gradio image to Torch Tensor [0,1]."""
    transform = transforms.Compose([
        transforms.Resize((32, 32)), 
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
    with torch.no_grad():
        output = model(tensor)
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_idx = output.max(1, keepdim=True)[1].item()
    
    # create dict for Gradio label component
    confidences = {LABELS[i]: float(probs[i]) for i in range(10)}
    return pred_idx, confidences

def amplify_noise(clean_tensor, adv_tensor):
    """Visualizes the noise by amplifying the difference."""
    noise = (adv_tensor - clean_tensor).abs()
    noise = noise * 50  # scale up noise visibility
    noise = torch.clamp(noise, 0, 1)
    return transforms.ToPILImage()(noise.squeeze().cpu())

# main demo function
def run_autoattack_demo(image, epsilon, robust_model_choice, attack_choice):
    if image is None:
        return None, None, None, None, None

    # prepare data
    image_width, image_height = image.size
    clean_tensor = preprocess(image)

    robust_model_name = MODEL_CHOICES[robust_model_choice]
    robust_model = load_robust_model(robust_model_name)

    # get ground truth
    pred_idx, _ = get_prediction(standard_model, clean_tensor)
    label_tensor = torch.tensor([pred_idx]).to(DEVICE)

    attack_type = ATTACK_CHOICES[attack_choice]

    # generate attack
    if epsilon == 0:
        adv_tensor = clean_tensor
    elif attack_type == "fgsm":
        adv_tensor = fgsm_attack(standard_model, clean_tensor, epsilon)
    else:
        try:
            adversary = AutoAttack(standard_model, norm='Linf', eps=epsilon, version='custom', verbose=False, device=DEVICE)
            adversary.attacks_to_run = [attack_type] 
            adv_tensor = adversary.run_standard_evaluation(clean_tensor, label_tensor, bs=1)
        except Exception as e:
            print(f"AutoAttack error: {e}")
            adv_tensor = fgsm_attack(standard_model, clean_tensor, epsilon)

    # get predictions
    # standard model on attacked image
    _, std_pred = get_prediction(standard_model, adv_tensor)
    # robust model on attacked image
    _, rob_pred = get_prediction(robust_model, adv_tensor)
    
    # visuals
    perturbed_img_display = transforms.ToPILImage()(adv_tensor.squeeze().cpu())
    noise_img_display = amplify_noise(clean_tensor, adv_tensor)

    perturbed_img_display = perturbed_img_display.resize((image_width, image_height), Image.NEAREST)
    noise_img_display = noise_img_display.resize((image_width, image_height), Image.NEAREST)
    
    return perturbed_img_display, noise_img_display, std_pred, rob_pred

# Gradio interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Adversarial Attack Demo (RobustBench)")
    gr.Markdown("""
    **Compare** a Standard ResNet against various Robust Models under different adversarial attacks.  
    """)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("""
            Please upload an image from one of these CIFAR-10 categories for best results:
            - **Airplane** | **Automobile** | **Bird** | **Cat** | **Deer** | **Dog** | **Frog** | **Horse** | **Ship** | **Truck**
            
            *The image will be automatically resized to 32×32 pixels.*
            """)

            input_img = gr.Image(label="Upload Image (Resized to 32x32)", type="pil")

            # model selection 
            model_dropdown = gr.Dropdown(
                choices=list(MODEL_CHOICES.keys()),
                value="Bartoldson2024 WRN-94-16", 
                label="Select Robust Model"
            )

            # attack selection
            attack_dropdown = gr.Dropdown(
                choices=list(ATTACK_CHOICES.keys()),
                value="FGSM (Fast)",
                label="Select Attack Method"
            )

            # AutoAttack usually uses eps=8/255 (approx 0.03) for CIFAR-10
            epsilon_slider = gr.Slider(0, 0.1, value=0.00, step=0.001, label="Attack Strength (Epsilon)")
            run_btn = gr.Button("Run AutoAttack", variant="primary")
        
        with gr.Column(scale=1):
            gr.Markdown("### Visualization")
            adv_image_out = gr.Image(label="Attacked Image (Adversarial)")
            noise_image_out = gr.Image(label="The Noise Pattern (Amplified)")
            
    gr.Markdown("### Model Predictions on Attacked Image")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### Standard Model")
            std_label = gr.Label(num_top_classes=10)
        
        with gr.Column():
            gr.Markdown("### Robust Model")
            rob_label = gr.Label(num_top_classes=10)

    # attack info
    gr.Markdown("""
    ### Attack Methods:
    - **FGSM**: Fast single-step attack (< 1 sec)
    - **APGD-CE/DLR**: Strong iterative attacks (2-5 sec)
    - **FAB**: Minimal perturbation attack (3-8 sec)
    - **Square**: Black-box attack (5-10 sec)
    - **APGD-T/FAB-T**: Targeted versions (slower)
    """)

    run_btn.click(
        fn=run_autoattack_demo,
        inputs=[input_img, epsilon_slider, model_dropdown, attack_dropdown],
        outputs=[adv_image_out, noise_image_out, std_label, rob_label]
    )

if __name__ == "__main__":
    demo.launch(debug=True, share=True)