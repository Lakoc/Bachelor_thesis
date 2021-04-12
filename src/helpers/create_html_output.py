from bs4 import BeautifulSoup
from outputs import outputs
from os.path import join
from helpers.dir_exist import create_if_not_exist


def load_template(template_path):
    with open(template_path, 'r') as file:
        soup = BeautifulSoup(file, "html.parser")
    return soup


def create_output(stats, template, path, file_name):
    html = load_template(template)

    create_if_not_exist(join(path, 'plots'))
    outputs.plot_speech_time_comparison(stats['speech_ratio'][0],
                                        f'{join(path, f"plots/{file_name}_speech_ratio")}.png')

    title = html.find(id='session_title')
    title.string.replace_with(title.text.replace('###', file_name))

    speech_comparison = html.find(id='figure-speech_ratio')
    speech_comparison['src'] = f'plots/{file_name}_speech_ratio.png'

    transcription = html.find(id='attachment-transcription')
    transcription['href'] = f'{file_name}.txt'
    transcription['download'] = f'{file_name}.txt'

    audio = html.find(id='attachment-audio')
    audio['href'] = f'{file_name}.wav'
    audio['download'] = f'{file_name}.wav'

    vad = html.find(id='attachment-vad')
    vad['href'] = f'{file_name}.rttm'
    vad['download'] = f'{file_name}.rttm'

    with open(f'{join(path, file_name)}.html', 'w') as output:
        output.write(str(html))
