from bs4 import BeautifulSoup


def load_template(template_path):
    with open(template_path, 'r') as file:
        soup = BeautifulSoup(file, "html.parser")
    return soup


def create_output(stats, template, path):
    html = load_template(template)

    with open(path, 'w') as output:
        output.write(str(html))
