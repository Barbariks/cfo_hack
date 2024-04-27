import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pickle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TextClassifier(nn.Module):
    def __init__(self, n_classes):
        super(TextClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('DeepPavlov/rubert-base-cased', return_dict=False)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        output = self.dropout(outputs[1])
        return self.classifier(output)

n_classes = 12
model = TextClassifier(n_classes);
model.load_state_dict(torch.load('model_CFO_sd.pth', map_location=device))
model.to(device)
model.eval()

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

def predict_url(description):
    encoded = tokenizer.encode_plus(
        description,
        add_special_tokens=True,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted_index = torch.max(outputs, dim=1)

    return le.inverse_transform([predicted_index.item()])[0]