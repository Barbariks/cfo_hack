import pandas as pd

courses = pd.read_csv('course_data.csv')

def get_course_data(course_link : str):
    info = courses.loc[courses['Ссылка на продукт'] == course_link].iloc[0]

    return {
        'parsedBody': {
            'url_vac': info['Ссылка на продукт'],
            'time': info['Длительность'],
            'cost': info['стоимость обучения'],
            'forma': info['Формат обучения'],
            'desc': info['Профессия'],
        }
    }