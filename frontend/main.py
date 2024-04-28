import requests
import streamlit as st
from PIL import Image
from pathlib import Path
import logging
import base64
import json
from matplotlib import pyplot as plt


# def get_project_root() -> str:
#     return str(Path(__file__).parent.parent.parent)


# def load_image(image_name: str) -> Image:
#     return Image.open(Path(get_project_root()) / f"references/{image_name}")


# st.sidebar.image(load_image("img/logo.png"), use_column_width=True)
# with st.sidebar:
#     selected = option_menu('SimiLo', ["Intro", 'Search','About'],
#         icons=['play-btn','search','info-circle'],menu_icon='intersect', default_index=0)

# import streamlit as st
# from streamlit_option_menu import option_menu

# # 2. horizontal menu
# selected2 = option_menu(None, ["Home", "Upload", "Tasks", 'Settings'], 
#     icons=['house', 'cloud-upload', "list-task", 'gear'], 
#     menu_icon="cast", default_index=0, orientation="horizontal")
# selected2
#background-image:  linear-gradient(#868be4 2px, transparent 2px), linear-gradient(90deg, #868be4 2px, transparent 2px), linear-gradient(#868be4 1px, transparent 1px), linear-gradient(90deg, #868be4 1px, #e9e9f8 1px);
#background-size: 50px 50px, 50px 50px, 10px 10px, 10px 10px;
#background-position: -2px -2px, -2px -2px, -1px -1px, -1px -1px;
#background-image: url("https://catherineasquithgallery.com/uploads/posts/2021-02/1614380674_69-p-fon-abstraktsiya-svetlii-geometriya-76.jpg");
#background-image: url("https://img.freepik.com/free-photo/glowing-spider-web-dark-abstract-backdrop-generated-by-ai_188544-36464.jpg?t=st=1714086465~exp=1714090065~hmac=eb294949183cf9872d61d6f464e6fd8cea816540d71eeaa88d2852888f97cb96&w=1380");
#background-image: url("https://understandbrain.com/wp-content/uploads/2023/09/glowing-spider-web-dark-abstract-backdrop-generated-by-ai.jpg");   
#background-image: url("https://thecharnelhouse.org/wp-content/uploads/2014/06/al-3-1926-lc3a1szlc3b3-moholy-nagy-oil-industrial-paints-and-pencil-on-aluminum-40-x-40-cm.jpg");
#color: #E8E6BB
#color: #DEFFA8;
page_bg_img = """
<style>

[data-testid="stAppViewContainer"] {  
    background-image: url("https://sun9-23.userapi.com/impg/XiO97rsW6bxg28kT5PH2JzV0j9St0JeHe8mWxQ/BSwgAokPBQ0.jpg?size=2560x1440&quality=96&sign=d30f4a2726161edc785e389ec67e5c2e&type=album");   
    # background-image: url("https://understandbrain.com/wp-content/uploads/2023/09/glowing-spider-web-dark-abstract-backdrop-generated-by-ai.jpg");   

    
    background-size: cover;

    # background-color: #ffffff;
    # opacity: 0.8;
    # background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #ffffff 350px ), repeating-linear-gradient( #444cf755, #444cf7 );

p { color: black;
background-color: black, transparent 50;
font-size: 20px;
# }
}
[data-testid="stHeader"]{                    
    background-color: rgba(0, 0, 0, 0);
    color: black;
    font-size: 100px;
    font_weight: bold;
    
}
[data-testid="stSidebar"]{                    
    background-color: black;
}

[data-testid="stButton"]{                    
    border-color: rgba(0, 0, 0, 0);
}

div[data-baseweb = "select"] > div{
    # background-color: black;
    # border-color: #CDFF6A;
    background-color: #DEFFA8;
    border-color: rgba(0, 0, 0, 0);
    border-radius: 10px;
    #color: #E8E6BB;
    color: black;
}
[data-testid = "stForm"]{
background: black;
}


[data-baseweb="icon"]{
#color: #DEFFA8;
color: black;
}

[class="st-emotion-cache-sy3zga e1gc5fo21"]{
color: #E8E6BB;
# background-color: red; 
}

[class="st-emotion-cache-9ycgxx e1b2p2ww12"]{
#color: #E8E6BB;
color: black;
#подпись drag and drop
}

[class="st-emotion-cache-7oyrr6 e1bju1570"]{
#color: #E8E6BB;
color: black;
#подпись лимит
}

[class="st-emotion-cache-19rxjzo ef3psqc12"]{
background-color: #DEFFA8;
color: black;
# border-color: #DEFFA8;
#browse file in drag and drop
}

[role="button"]{
background-color: #DEFFA8;
border-color: rgba(0,0,0,0);
border-radius: 10px;
}
[role="textbox"]{
background-color: #DEFFA8;
border-radius: 10px;
font-size: 20px;
# драг н дроп
}
[class="st-emotion-cache-wn8ljn e1b2p2ww13"]{
color: black;
# цвет иконки загрузки
}

[data-baseweb="input"]{
border-color: #DEFFA8;
}
[data-baseweb="base-input"]{
background-color: #DEFFA8;
color: black;
border-color: #DEFFA8;
-webkit-text-fill-color: black;
caret-color: red;
}
[data-baseweb="textarea"]{
border-color: #DEFFA8;
color: black;
caret-color: red;
}
[class="st-emotion-cache-1mdkfbq e1b2p2ww3"]{
color: black; #цвет иконки пдф
}

[class="st-emotion-cache-1lp7pgu ef3psqc9"]{
color: black; #цвет крестика
}



[class="stFileUploaderFileData st-emotion-cache-1l4firl e1b2p2ww7"]{
color: black; #вет загруженного пдф
font-size: 20px;
}

# [class="st-emotion-cache-7oyrr6 e1bju1570"]{
# color: #3A0603;
# }


.st-cc > div > div > div > input {
    border-color: rgba(0, 0, 0, 0);
    #border-width: 2px;
    #border-style: solid;
    border-radius: 5px;
    
}

.stTextInput > div > div > input {
    background-color: #DEFFA8;
    color: black;
}



</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

gradient_text_html = """
<style>
.gradient-text {
    # background-color: #CDFF6A;
    # background-color: #3A0603;
    background-color: black;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 46px;
    font-family: 'Courier New', Courier, monospace;
    # font-family: Franklin Gothic;
    font-weight: bold;
    padding: 10px;
}

</style>
<div class="gradient-text"><Найди профессию мечты/></div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)
#options = ['Пройти опрос', 'Посмотреть результат']

#file = st.sidebar.radio('Выберите:', options)

# if st.button("Browse", className="file-upload-btn"):
#     uploaded_file = st.file_uploader("Choose a file")
#     if uploaded_file is not None:
#         # Process the uploaded file
#         st.write("File uploaded successfully!")

def extract_data(uploaded_file):
    file_content = uploaded_file.read()
    # st.write(file_content)
    encoded_file_content = base64.b64encode(file_content).decode('utf-8')
    # st.write(encoded_file_content)
    return {
        "file_content": encoded_file_content
    }

url= ""
desc= ""
long= ""
cost= ""
forma= ""


def output(data):
            url = data["parsedBody"]["url_vac"]   #ссылка на курс
            url = str(url)

            long = data["parsedBody"]["time"]  #длительность обучения
            long = str(long) + '.'

            cost = data["parsedBody"]["cost"]   #стоимость
            cost = str(cost)

            forma = data["parsedBody"]["forma"] #формат обучения
            forma = str(forma) + '.'

            desc = data["parsedBody"]["desc"]   #описание
            desc = str(desc) + '.'

            prof = data["parsedBody"]["professions"]  # профессии
            num_prof = data["parsedBody"]["num_professions"]  # профессии

            st.write("Ссылка на курс: " + url)

            st.write("Длительность курса: " + long)

            st.write("Стоимость обучения от: " + cost)

            st.write("Формат обучения: " + forma)


            bold_text = """
            <style>
            .bold-text{
            color:black;
            font-weight: bold;
            font-size: 20px;
            }
            </style>
            <div class = "bold-text">Описание курса</div>
            """

            st.markdown(bold_text, unsafe_allow_html=True)
            st.write(desc)
            fig = plt.figure()
            plt.bar(prof, num_prof)
            plt.xticks(rotation=45)
            st.write(fig)
            otziv = """
            <style>
            .otziv{
                color:black;
                font-weight: bold;
                font-size: 20px;
            }
            </style>
            <div class = "otziv">Оцените нашу работу</div>
            """
            st.markdown(otziv, unsafe_allow_html=True)
            otziv_polz = st.text_input('Напишите отзыв', help='Напишите отзыв', key = 'text')
            print(otziv_polz)
            ot = {
                "otziv_polz": otziv_polz
            }
            ot_pol = requests.post('https://echo.free.beeceptor.com', json=ot)
            print(ot_pol.text)
            thx = """
                <style>
                .thx{
                color:black;
                font-weight: bold;
                font-size: 20px;
                }
                </style>
                <div class = "thx">Спасибо за отзыв!</div>
                """
            st.markdown(thx, unsafe_allow_html=True)

                    
            




select = st.selectbox(
        'Выберите формат добавления вакансии',
        ('Загрузить PDF','Вставить ссылку на вакансию', 'Ввести текстовое описание вакансии'))
if select == 'Загрузить PDF':
    uploaded_file = st.file_uploader('Перетащите PDF файл', type="pdf")
    if uploaded_file is not None:
        df = extract_data(uploaded_file)
        if st.button("Найти по PDF"):
            data_to_send = df
            response = requests.post('http://127.0.0.1:8000/process_pdf', json=data_to_send)

            data = response.json()
            output(data)
            


if select == 'Вставить ссылку на вакансию':
    url_vac = st.text_input('Вставьте ссылку на вакансию', key="text_input", help='Вставьте сюда ссылку')
    if st.button("Найти по ссылке"):
        data_to_send = {
            "url_vac": url_vac
        }
        response = requests.post('http://127.0.0.1:8000/process_url', json=data_to_send)
        print(response.text)

        data = response.json()
        output(data)

if select == 'Ввести текстовое описание вакансии':
    text_desc_vac = st.text_area('Введите текстовое описание вакансии', key="text_input", help='Введите сюда описание',height=150)
    if st.button("Найти по описанию"):
        data_to_send = {
            "text_vac": text_desc_vac
        }
        response = requests.post('http://127.0.0.1:8000/process_text', json=data_to_send)

        data = response.json()
        output(data)


# def output(data):
#             url = data["/*имясписка*/"]["url_vac"]   #ссылка на курс
#             url = str(url)

#             long = data["parsedBody"]["time"]  #длительность обучения
#             long = str(long)

#             cost = data["parsedBody"]["cost"]   #стоимость
#             cost = str(cost)

#             forma = data["parsedBody"]["forma"] #формат обучения
#             forma = str(forma)

#             desc = data["parsedBody"]["desc"]   #описание
#             desc = str(desc)

#             st.write("Ссылка на курс: " + url)

#             st.write("Длительность курса " + long)

#             st.write("Стоимость обучения от " + cost +" ₽")

#             st.write("Формат обучения: " + forma)


#             bold_text = """
#             <style>
#             .bold-text{
#             color:black;
#             font-weight: bold;
#             font-size: 20px;
#             }
#             </style>
#             <div class = "bold-text">Описание курса</div>
#             """

#             st.markdown(bold_text, unsafe_allow_html=True)
#             st.write(desc)

#             otziv = """
#             <style>
#             .otziv{
#                 color:black;
#                 font-weight: bold;
#                 font-size: 20px;
#             }
#             </style>
#             <div class = "otziv">Оцените нашу работу</div>
#             """
#             st.markdown(otziv, unsafe_allow_html=True)

#             otziv_polz = st.text_area('Напишите отзыв', help='Напишите отзыв',height=100)
#             if st.button("Отправить отзыв"):
#                 data_to_send = {
#                     "otziv_polz": otziv_polz
#                 }
#                 response = requests.post('https://echo.free.beeceptor.com', json=data_to_send)
#                 print(response.text)
#                 thx = """
#                     <style>
#                     .thx{
#                     color:black;
#                     font-weight: bold;
#                     font-size: 20px;
#                     }
#                     </style>
#                     <div class = "thx">Спасибо за отзыв!</div>
#                     """
#                 st.markdown(thx, unsafe_allow_html=True)
