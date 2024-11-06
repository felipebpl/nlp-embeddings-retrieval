import os
from bs4 import BeautifulSoup
import requests

# URL principal com a lista de ensaios
url = 'http://www.paulgraham.com/articles.html'
soup = BeautifulSoup(requests.get(url).text, "html.parser")
essay_links = soup.find_all('a')

def make_txt(content, file_name):
    """
    Salva o conteúdo formatado em um arquivo .txt.
    """
    path = 'essays/'
    if not os.path.exists(path):
        os.makedirs(path)
    sanitized_file_name = file_name.replace('/', '_').strip()
    with open(f"{path}{sanitized_file_name}.txt", 'w', encoding='utf-8') as f:
        f.write(content)

def extract_text(element):
    """
    Extrai o texto de um elemento BeautifulSoup, corrigindo a formatação básica (itálico, negrito),
    sem duplicar ou adicionar elementos desnecessários.
    """
    parts = []
    for item in element.descendants:
        if item.name == 'br':
            parts.append('\n')
        elif item.name in ['i', 'em']:
            text = f"{item.get_text()}"
            if not parts or parts[-1] != text:
                parts.append(text)
        elif item.name in ['b', 'strong']:
            text = f"{item.get_text()}"
            if not parts or parts[-1] != text:
                parts.append(text)
        elif item.name == 'a':
            href = item.get('href')
            link_text = f"{item.get_text()}" if href else item.get_text()
            if not parts or parts[-1] != link_text:
                parts.append(link_text)
        elif isinstance(item, str):
            if not parts or parts[-1] != item:
                parts.append(item)
    return ''.join(parts).strip()

for i in range(4, len(essay_links) - 1):
    essay_url = "http://www.paulgraham.com/" + str(essay_links[i]['href'])
    link = BeautifulSoup(requests.get(essay_url).text, "html.parser")
    content = ""
    for font_tag in link.find_all('font', {"face": "verdana", "size": "2"}):
        content += extract_text(font_tag) + "\n\n"
    make_txt(content, str(essay_links[i].get_text()).strip())
    print(f"Saved: {essay_links[i].get_text().strip()}")