import numpy as np

import outputs.plots as plots
from os.path import join

import params
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
                        (stats['volume_changes'][0], stats['interruptions'][0], stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}_therapist_volume")}.png'), html,
                        'therapist_volume', file_name)

    create_plot_and_set(plots.plot_volume_changes,
                        (stats['volume_changes'][1], stats['interruptions'][1], stats['signal']['len'],
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

    create_plot_and_set(plots.plot_cross_talk_histogram,
                        (stats['interruptions'][0], stats['interruptions'][1], 10,
                         f'{join(path, f"plots/{file_name}_therapist_interruptions")}.png'), html,
                        'therapist_interruptions', file_name)

    create_plot_and_set(plots.plot_cross_talk_histogram,
                        (stats['interruptions'][1], stats['interruptions'][0], 10,
                         f'{join(path, f"plots/{file_name}_client_interruptions")}.png'), html,
                        'client_interruptions', file_name)

    create_plot_and_set(plots.plot_reaction_time_comparison,
                        (stats['reactions'][0], stats_overall['reactions']['therapist'],
                         f'{join(path, f"plots/{file_name}_reaction_time_therapist")}.png'), html,
                        'reaction_time_therapist', file_name)

    create_plot_and_set(plots.plot_reaction_time_comparison,
                        (stats['reactions'][1], stats_overall['reactions']['client'],
                         f'{join(path, f"plots/{file_name}_reaction_time_client")}.png'), html,
                        'reaction_time_client', file_name)

    create_plot_and_set(plots.plot_speech_bounds_len,
                        (stats['speech_joined'][0],
                         f'{join(path, f"plots/{file_name}_bounds_len_therapist")}.png'), html,
                        'bounds_len_therapist', file_name)

    create_plot_and_set(plots.plot_speech_bounds_len,
                        (stats['speech_joined'][1],
                         f'{join(path, f"plots/{file_name}_bounds_len_client")}.png'), html,
                        'bounds_len_client', file_name)

    create_plot_and_set(plots.plot_speed,
                        (stats['speed'][0], 8, stats['signal']['len'], f'{join(path, f"plots/{file_name}_speed_therapist")}.png'), html,
                        'speed_therapist', file_name)

    create_plot_and_set(plots.plot_speed,
                        (stats['speed'][1], 8,stats['signal']['len'], f'{join(path, f"plots/{file_name}_speed_client")}.png'), html,
                        'speed_client', file_name)

    create_plot_and_set(plots.plot_hesitations,
                        (stats['hesitations'][0], stats['signal']['len'], stats_overall['hesitations']['therapist'],
                         f'{join(path, f"plots/{file_name}_hesitations_therapist")}.png'), html,
                        'hesitations_therapist', file_name)

    create_plot_and_set(plots.plot_hesitations,
                        (stats['hesitations'][1], stats['signal']['len'], stats_overall['hesitations']['client'],
                         f'{join(path, f"plots/{file_name}_hesitations_client")}.png'), html,
                        'hesitations_client', file_name)


def add_volume_table(html, stats):
    """Add table of most energetic segments with theirs annotations"""
    t_body = html.find(id='volume_text_table').find("tbody")
    loud1_sorted = sorted(stats['loudness'][0], key=lambda l: stats['loudness'][0][l]['ratio'])
    loud2_sorted = sorted(stats['loudness'][1], key=lambda l: stats['loudness'][1][l]['ratio'])
    loud1_sorted = loud1_sorted[0: min(len(loud1_sorted), 10)]
    loud2_sorted = loud2_sorted[0: min(len(loud2_sorted), 10)]

    loudness = {key: stats['loudness'][0][key] for key in loud1_sorted}, {key: stats['loudness'][1][key] for key in
                                                                          loud2_sorted}
    for num, speaker in enumerate(loudness):
        for segment in speaker:
            text = speaker[segment]['text']
            ratio = speaker[segment]['ratio']
            volume = speaker[segment]['energy']
            if text is not None and ratio > 0:
                new_row = html.new_tag("tr")
                time_start_tag = html.new_tag('td')
                time_start_tag.string = f'{int(segment[0] * params.window_stride / 60)} m {int(segment[0] * params.window_stride % 60)} s'
                new_row.append(time_start_tag)

                time_dur_tag = html.new_tag('td')
                time_dur_tag.string = f'{(segment[1] - segment[0]) * params.window_stride:.2f}'
                new_row.append(time_dur_tag)

                speaker_tag = html.new_tag('td')
                speaker_tag.string = 'Terapeut' if num == 0 else 'Klient'
                new_row.append(speaker_tag)

                volume_tag = html.new_tag('td')
                volume_tag.string = f'{volume:.2f}'
                new_row.append(volume_tag)

                text_tag = html.new_tag('td')
                text_tag.string = text
                new_row.append(text_tag)

                t_body.append(new_row)


def add_texts(html, stats, stats_overall, texts, file_name):
    """Add texts from json to template html file"""
    title = html.find(id='session_title')
    title.string.replace_with(title.text.replace('###', file_name))

    html.find(id='speech_ratio_current').string = return_text_by_bounds(texts['speech_ratio']['current'],
                                                                        stats['speech_ratio'][0][0], [0.5, 0.2])

    html.find(id='speech_ratio_overall').string = return_text_by_bounds(texts['speech_ratio']['overall'],
                                                                        stats['speech_ratio'][0][0], stats_overall[
                                                                            'speech_ratio'])
    reaction_time = np.mean(stats["reactions"][0][:, 1] - stats["reactions"][0][:, 0]) * params.window_stride

    html.find(
        id='reaction_time_therapist').string = f'Průměrná reakční doba terapeuta v průběhu sezení je ' \
                                               f'{reaction_time:.2f} s. Průmerná hodnota mezi sezeními se pohybuje ' \
                                               f'kolem {stats_overall["reactions"]["therapist"][0]:.2f} ' \
                                               f'(+-{stats_overall["reactions"]["therapist"][1]:.2f}) s'

    reaction_time = np.mean(stats["reactions"][1][:, 1] - stats["reactions"][1][:, 0]) * params.window_stride

    html.find(
        id='reaction_time_client').string = f'Průměrná reakční doba klienta v průběhu sezení je ' \
                                            f'{reaction_time:.2f} s. Průmerná hodnota mezi sezeními se pohybuje ' \
                                            f'kolem {stats_overall["reactions"]["client"][0]:.2f} ' \
                                            f'(+-{stats_overall["reactions"]["client"][1]:.2f}) s'

    html.find(
        id='hesitations_therapist').string = f'Průměrný počet váhaní za minutu je ' \
                                             f'{stats_overall["hesitations"]["therapist"][0]:.2f}. Terapeut v ' \
                                             f'průběhu aktuálního sezení váhá průměrně ' \
                                             f'{stats["hesitations"][0].shape[0] / stats["signal"]["len"] * 60:.2f} ' \
                                             f'krát za minutu.'

    html.find(
        id='hesitations_client').string = f'Průměrný počet váhaní za minutu je ' \
                                          f'{stats_overall["hesitations"]["client"][0]:.2f}. Klient v ' \
                                          f'průběhu aktuálního sezení váhá průměrně ' \
                                          f'{stats["hesitations"][1].shape[0] / stats["signal"]["len"] * 60:.2f} ' \
                                          f'krát za minutu.'


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
    add_volume_table(html, stats)

    with open(f'{join(path, file_name)}.html', 'w') as output:
        output.write(str(html))
