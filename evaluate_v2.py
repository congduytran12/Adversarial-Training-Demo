import torch
from robustbench.data import load_cifar10
from robustbench.utils import load_model
from autoattack import AutoAttack
import torch.nn.functional as F
import csv
from torch.ultis.data import DataLoader, TensorDataset
from tqdm import tqdm

# setup and configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Running evaluation on {DEVICE}...")

# load test data
x_test, y_test = load_cifar10(n_examples=1000)
x_test, y_test = x_test.to(DEVICE), y_test.to(DEVICE)

# models to evaluate
MODELS = {
    "Standard": "Standard",
    "Bartoldson2024 WRN-94-16": "Bartoldson2024Adversarial_WRN-94-16",
    "Amini2024 MeanSparse WRN-94-16": "Amini2024MeanSparse_S-WRN-94-16",
    "Bartoldson2024 WRN-82-8": "Bartoldson2024Adversarial_WRN-82-8",
}

# attacks to evaluate
ATTACKS = {
    "FGSM": "fgsm",
    "APGD-CE": "apgd-ce",
    "APGD-DLR": "apgd-dlr",
    "FAB": "fab",
    "Square": "square",
    "APGD-T": "apgd-t",
    "FAB-T": "fab-t"
}

def fgsm_attack(model, images, labels, epsilon=8/255):
    """Performs Fast Gradient Sign Method (FGSM) attack."""
    images_adv = images.clone().detach()
    images_adv.requires_grad = True
    
    # forward pass
    outputs = model(images_adv)

    # calculate loss (cross-entropy)
    loss = F.cross_entropy(outputs, labels)
    
    # zero gradients
    model.zero_grad()

    # backward pass
    loss.backward()
    
    # collect data gradient
    data_grad = images_adv.grad.data

    # create the perturbed image
    perturbed_images = images_adv + epsilon * data_grad.sign()

    # clamp to maintain [0, 1] range
    perturbed_images = torch.clamp(perturbed_images, 0, 1)
    
    return perturbed_images

BATCH_SIZE = 32

# def evaluate_model(model, x_test, y_test, attack_name, attack_type, epsilon=8/255):
#     """Evaluate a model on a specific attack."""
#     model.eval()

#     if attack_type == "fgsm":
#         x_adv = fgsm_attack(model, x_test, y_test, epsilon)
#     else:
#         try:
#             adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='custom', verbose=False, device=DEVICE)
#             adversary.attacks_to_run = [attack_type]
#             if attack_type in ['apgd-ce', 'apgd-dlr']:
#                 adversary.apgd.n_restarts = 1
#             x_adv = adversary.run_standard_evaluation(x_test, y_test)
#         except Exception as e:
#             print(f"  Error with {attack_name}: {e}")
#             return None
    
#     # calculate accuracy
#     with torch.no_grad():
#         outputs = model(x_adv)
#         _, predicted = torch.max(outputs, 1)
#         correct = (predicted == y_test).sum().item()
#         accuracy = 100 * correct / len(y_test)
    
#     return accuracy

def evaluate_model(model, x_test, y_test, attack_name, attack_type, epsilon=8/255, batch_size=BATCH_SIZE):
    """Evaluate a model on a specific attack using minibatches to avoid OOM."""
    model.eval()
    dataset = TensorDataset(x_test, y_test)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_correct = 0
    total = 0

    if attack_type == "fgsm":
        # FGSM per-batch
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            xb_adv = fgsm_attack(model, xb, yb, epsilon)
            with torch.no_grad():
                outs = model(xb_adv)
                _, preds = torch.max(outs, 1)
                total_correct += (preds == yb).sum().item()
                total += yb.size(0)
            del xb_adv, outs, preds
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        try:
            adversary = AutoAttack(model, norm='Linf', eps=epsilon, version='custom', verbose=False, device=DEVICE)
            adversary.attacks_to_run = [attack_type]
            if attack_type in ['apgd-ce', 'apgd-dlr']:
                adversary.apgd.n_restarts = 1

            # run attack per-batch
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                xb_adv = adversary.run_standard_evaluation(xb, yb, bs=xb.size(0))
                with torch.no_grad():
                    outs = model(xb_adv)
                    _, preds = torch.max(outs, 1)
                    total_correct += (preds == yb).sum().item()
                    total += yb.size(0)
                del xb_adv, outs, preds
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Error with {attack_name}: {e}")
            return None

    accuracy = 100.0 * total_correct / total if total > 0 else 0.0
    return accuracy

# run evaluation
print("\n" + "="*80)
print("Adversarial Robustness Evaluation")
print("="*80)
print(f"Test samples: {len(y_test)}")
print(f"Epsilon: 8/255")
print("="*80 + "\n")

results = {}

for model_name, model_id in MODELS.items():
    print(f"\n{'='*80}")
    print(f"Evaluating Model: {model_name}")
    print(f"{'='*80}")
    
    # load model
    try:
        model = load_model(model_name=model_id, dataset='cifar10', threat_model='Linf').to(DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model {model_name}: {e}")
        continue
    
    # clean accuracy
    with torch.no_grad():
        outputs = model(x_test)
        _, predicted = torch.max(outputs, 1)
        correct = (predicted == y_test).sum().item()
        clean_acc = 100 * correct / len(y_test)
    
    print(f"Clean Accuracy: {clean_acc:.2f}%")
    
    results[model_name] = {"Clean": clean_acc}
    
    # evaluate on each attack
    for attack_name, attack_type in tqdm(ATTACKS.items(), desc=""):
        print(f"  Testing {attack_name}...", end=" ", flush=True)
        acc = evaluate_model(model, x_test, y_test, attack_name, attack_type)
        if acc is not None:
            results[model_name][attack_name] = acc
            print(f"Robust Accuracy: {acc:.2f}%")
        else:
            print("Failed")
    
    # free memory
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# print summary
print("\n" + "="*80)
print("SUMMARY - Robust Accuracy (%) on Adversarial Examples")
print("="*80)
print(f"{'Model':<40} {'Clean':<8}", end="")
for attack_name in ATTACKS.keys():
    print(f"{attack_name:<12}", end="")
print()
print("-"*80)

for model_name, accs in results.items():
    print(f"{model_name:<40} {accs.get('Clean', 0):<8.2f}", end="")
    for attack_name in ATTACKS.keys():
        if attack_name in accs:
            print(f"{accs[attack_name]:<12.2f}", end="")
        else:
            print(f"{'N/A':<12}", end="")
    print()

print("="*80)

with open('evaluation_result.csv', 'w', newline='') as csvfile:
    fieldnames = ['Model', 'Clean'] + list(ATTACKS.keys())
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    for model_name, accs in results.items():
        row = {'Model': model_name, 'Clean': f"{accs.get('Clean', 0):.2f}"}
        for attack_name in ATTACKS.keys():
            if attack_name in accs:
                row[attack_name] = f"{accs[attack_name]:.2f}"
            else:
                row[attack_name] = 'N/A'
        writer.writerow(row)

print("\nResult saved to evaluation_result.csv")
print("="*80)
