import outputs.plots as plots
from os.path import join
from helpers.dir_exist import create_if_not_exist
from shutil import copyfile
from helpers.plot_helpers import word_cloud

from helpers.load_files import load_template


def return_text_by_bounds(options, num, overall):
    """Find to which group value belongs"""
    keys = list(options.keys())
    key = keys[1]
    if num < overall[0] - overall[1]:
        key = keys[0]
    elif num > overall[0] + overall[1]:
        key = keys[2]
    return options[key]


def create_plot_and_set(fun, args, html, element_id, file_name):
    """Plot to file and set that path to html as source"""
    fun(*args)
    html.find(id=f'figure-{element_id}')['src'] = f'plots/{file_name}_{element_id}.png'


def add_plots(html, stats, stats_overall, path, file_name):
    """Create plot and set path in html file"""
    create_if_not_exist(join(path, 'plots'))

    create_plot_and_set(plots.plot_speech_time_comparison,
                        (stats['speech_ratio'][0], f'{join(path, f"plots/{file_name}_speech_ratio")}.png'), html,
                        'speech_ratio', file_name)

    create_plot_and_set(plots.plot_speech_time_comparison_others,
                        (stats['speech_ratio'][0][0], stats_overall['speech_ratio'],
                         f'{join(path, f"plots/{file_name}_speech_ratio_overall")}.png'), html,
                        'speech_ratio_overall', file_name)

    create_plot_and_set(plots.plot_volume_changes,
                        (stats['volume_changes'][0], stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}_therapist_volume")}.png'), html,
                        'therapist_volume', file_name)

    create_plot_and_set(plots.plot_volume_changes,
                        (stats['volume_changes'][1], stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}_client_volume")}.png'), html,
                        'client_volume', file_name)

    create_plot_and_set(word_cloud,
                        (stats['texts'][0],
                         f'{join(path, f"plots/{file_name}_therapist_cloud")}.png'), html,
                        'therapist_cloud', file_name)

    create_plot_and_set(word_cloud,
                        (stats['texts'][1],
                         f'{join(path, f"plots/{file_name}_client_cloud")}.png'), html,
                        'client_cloud', file_name)


def add_texts(html, stats, stats_overall, texts, file_name):
    """Add texts from json to template html file"""
    title = html.find(id='session_title')
    title.string.replace_with(title.text.replace('###', file_name))

    html.find(id='speech_ratio_current').string = return_text_by_bounds(texts['speech_ratio']['current'],
                                                                        stats['speech_ratio'][0][0], [0.5, 0.2])

    html.find(id='speech_ratio_overall').string = return_text_by_bounds(texts['speech_ratio']['overall'],
                                                                        stats['speech_ratio'][0][0], stats_overall[
                                                                            'speech_ratio'])


def add_attachments(html, file_name):
    """Add attachments to end of the file"""
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
    """Create single paged html document with overall statistics of session from provided template"""
    html = load_template(template)
    copyfile(f'{template.split(".html")[0]}.css', f'{path}/style.css')

    html.find(rel="stylesheet")['href'] = 'style.css'

    add_plots(html, stats, stats_overall, path, file_name)
    add_texts(html, stats, stats_overall, texts, file_name)
    add_attachments(html, file_name)

    with open(f'{join(path, file_name)}.html', 'w') as output:
        output.write(str(html))
