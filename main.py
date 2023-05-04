# Import libraries
import torch
class Net(torch.nn.Module):
    def __init__(self):
        super().__init__()        
        self.fc1 = torch.nn.Linear(in_features=1, out_features=1)
    
     

    def forward(self, x):
      out = self.fc1(x)    
      return out
    
# Define inference function
def inference(model: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
    """Inference function.

    Args:
        model (torch.nn.Module): Model.
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Output tensor.
    """
    # Set model to evaluation mode
    model.eval()
    # Disable gradient computation
    with torch.no_grad():
        # Forward pass
        y = model(x)
    # Return output
    return y

# Load model
model = Net()

model.load_state_dict(torch.load('model.pt'))

# # Input tensor
x = torch.tensor([[1.0], [2.0], [3.0]])
# # Inference
y = inference(model, x)

# # Print output
print(y)