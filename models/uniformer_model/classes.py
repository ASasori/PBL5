from modelxs_infer import uniformer_xxs
import torch
import torch.nn.functional as F
import os

def load_model():
    module_dir = os.path.dirname(os.path.abspath(__file__))

    state_dict = torch.load(os.path.join(module_dir,'uniformer_xxs32_160_k400.pth'), map_location='cpu')
    xxs_model = uniformer_xxs()
    xxs_model.load_state_dict(state_dict)
    for param in xxs_model.parameters():
        param.requires_grad = False

    class UniformerXXSFinetune(torch.nn.Module):
        def __init__(self, out_class=20):
            super(UniformerXXSFinetune, self).__init__()
            self.pretrained = xxs_model
            self.fc = torch.nn.Linear(400,out_class)

        def forward(self, x):
            x = self.pretrained(x)[0]
            x = self.fc(x)
            return F.softmax(x,dim=-1)

    # Instantiate the model
    model = UniformerXXSFinetune()

    model.load_state_dict(torch.load(os.path.join(module_dir,'xxs_finetuning_5_checkpoint_epoch_5.pth'),map_location='cpu'))
    model.eval()
    return model

