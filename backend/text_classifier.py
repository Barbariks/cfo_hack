import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re

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
model = TextClassifier(n_classes)
model.load_state_dict(torch.load('model_CFO_sd.pth', map_location=device))
model.to(device)
model.eval()

with open('label_encoder.pkl', 'rb') as file:
    le = pickle.load(file)

def clean_text(text):
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return ' '.join(words)

nltk.download('stopwords')
stop_words = set(stopwords.words('russian'))

tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')

datasetTMP = pd.read_excel('datasetTMP.xlsx')
datasetTMP = datasetTMP.drop(columns="Название продукта")

tfidf = TfidfVectorizer()
tfidf_matrix = tfidf.fit_transform(datasetTMP['Описание'])

def recommend_product(user_description, dataset):
    tfidf_desc = TfidfVectorizer()
    tfidf_matrix_desc = tfidf_desc.fit_transform(dataset["Описание"])
    cleaned_query = clean_text(user_description)
    query_vector_desc = tfidf_desc.transform([cleaned_query])
    cos_sim_desc = cosine_similarity(query_vector_desc, tfidf_matrix_desc)
    enhanced_cos_sim = cos_sim_desc * 2  # Увеличиваем влияние TF-IDF
    best_idx = enhanced_cos_sim.argmax()
    return dataset.iloc[best_idx]['Ссылка на продукт']

def predict_url(description):
    encoded = tokenizer.encode_plus(
        description, add_special_tokens=True, max_length=512,
        padding='max_length', truncation=True, return_attention_mask=True,
        return_tensors='pt')
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        _, predicted_index = torch.max(outputs, dim=1)
    return le.inverse_transform([predicted_index.item()])[0]

def ensemble_predict(description):
    predicted_url = predict_url(description)
    recommended_product = recommend_product(description, datasetTMP)

    return predicted_url if predicted_url == recommended_product else recommended_product