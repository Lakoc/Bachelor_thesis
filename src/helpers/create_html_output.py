from bs4 import BeautifulSoup

import outputs.plots as plots
from os.path import join
from helpers.dir_exist import create_if_not_exist
from shutil import copyfile


def return_text_by_bounds(options, num, left, right):
    keys = list(options.keys())
    key = keys[2]
    if num < left:
        key = keys[0]
    elif left <= num <= right:
        key = keys[1]
    return options[key]


def load_template(template_path):
    with open(template_path, 'r') as file:
        soup = BeautifulSoup(file, "html.parser")
    return soup


def add_plots(html, stats, stats_overall, path, file_name):
    create_if_not_exist(join(path, 'plots'))
    plots.plot_speech_time_comparison(stats['speech_ratio'][0],
                                      f'{join(path, f"plots/{file_name}_speech_ratio")}.png')
    html.find(id='figure-speech_ratio')['src'] = f'plots/{file_name}_speech_ratio.png'

    plots.plot_speech_time_comparison_others(stats['speech_ratio'][0][0], stats_overall['speech_ratio'],
                                             f'{join(path, f"plots/{file_name}_speech_ratio_overall")}.png')
    html.find(id='figure-speech_ratio_overall')['src'] = f'plots/{file_name}_speech_ratio_overall.png'


def add_texts(html, stats, stats_overall, texts, file_name):
    title = html.find(id='session_title')
    title.string.replace_with(title.text.replace('###', file_name))

    html.find(id='speech_ratio_current').string = return_text_by_bounds(texts['speech_ratio']['current'],
                                                                        stats['speech_ratio'][0][0], 0.3, 0.7)

    html.find(id='speech_ratio_overall').string = return_text_by_bounds(texts['speech_ratio']['overall'],
                                                                        stats['speech_ratio'][0][0] - stats_overall[
                                                                            'speech_ratio'], -0.2, 0.2)


def add_attachments(html, file_name):
    transcription = html.find(id='attachment-transcription')
    transcription['href'] = f'{file_name}.txt'
    transcription['download'] = f'{file_name}.txt'

    audio = html.find(id='attachment-audio')
    audio['href'] = f'{file_name}.wav'
    audio['download'] = f'{file_name}.wav'

    vad = html.find(id='attachment-vad')
    vad['href'] = f'{file_name}.rttm'
    vad['download'] = f'{file_name}.rttm'


def create_output(stats, stats_overall, texts, template, path, file_name):
    html = load_template(template)
    copyfile(f'{template.split(".html")[0]}.css', f'{path}/style.css')

    html.find(rel="stylesheet")['href'] = 'style.css'

    add_plots(html, stats, stats_overall, path, file_name)
    add_texts(html, stats, stats_overall, texts, file_name)
    add_attachments(html, file_name)

    with open(f'{join(path, file_name)}.html', 'w') as output:
        output.write(str(html))
