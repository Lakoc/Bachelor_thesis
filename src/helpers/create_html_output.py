import numpy as np

from outputs import plots
from os.path import join
import pickle
import params
from helpers.plot_helpers import word_cloud

from helpers.load_files import load_template
from helpers.dir_exist import create_if_not_exist


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
    html.find(id=f'figure-{element_id}')['src'] = f'plots/{file_name}/{element_id}.png'


def add_plots(html, stats, stats_overall, path, file_name):
    """Create plot and set path in html file"""
    create_if_not_exist(join(path, f'plots/{file_name}'), verbose=False)

    create_plot_and_set(plots.speech_time_comparison,
                        (stats['speech_ratio'][0], f'{join(path, f"plots/{file_name}/speech_ratio")}.png'),
                        html,
                        'speech_ratio', file_name)

    create_plot_and_set(plots.silence_ratio,
                        (
                            stats['speech_ratio'][1][0],
                            f'{join(path, f"plots/{file_name}/silence_ratio_therapist")}.png'),
                        html,
                        'silence_ratio_therapist', file_name)

    create_plot_and_set(plots.silence_ratio,
                        (stats['speech_ratio'][1][1], f'{join(path, f"plots/{file_name}/silence_ratio_client")}.png'),
                        html,
                        'silence_ratio_client', file_name)

    create_plot_and_set(plots.speech_time_comparison_others,
                        (stats['speech_ratio'][0][0], stats_overall['speech_ratio']['therapist'],
                         f'{join(path, f"plots/{file_name}/speech_ratio_overall")}.png'), html,
                        'speech_ratio_overall', file_name)

    create_plot_and_set(plots.volume_changes,
                        (stats['volume_changes'][0], stats['interruptions'][0], stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}/therapist_volume")}.png'), html,
                        'therapist_volume', file_name)

    create_plot_and_set(plots.volume_changes,
                        (stats['volume_changes'][1], stats['interruptions'][1], stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}/client_volume")}.png'), html,
                        'client_volume', file_name)

    create_plot_and_set(word_cloud,
                        (stats['texts'][0],
                         f'{join(path, f"plots/{file_name}/therapist_cloud")}.png'), html,
                        'therapist_cloud', file_name)

    create_plot_and_set(word_cloud,
                        (stats['texts'][1],
                         f'{join(path, f"plots/{file_name}/client_cloud")}.png'), html,
                        'client_cloud', file_name)

    create_plot_and_set(plots.interruptions_histogram,
                        (stats['interruptions_len'][0], stats_overall['interruptions_len']['therapist'],
                         f'{join(path, f"plots/{file_name}/therapist_interruptions")}.png'), html,
                        'therapist_interruptions', file_name)

    create_plot_and_set(plots.interruptions_histogram,
                        (stats['interruptions_len'][1], stats_overall['interruptions_len']['client'],
                         f'{join(path, f"plots/{file_name}/client_interruptions")}.png'), html,
                        'client_interruptions', file_name)

    create_plot_and_set(plots.reaction_time_comparison,
                        (stats['reactions'][0], stats_overall['reactions']['therapist'],
                         f'{join(path, f"plots/{file_name}/reaction_time_therapist")}.png'), html,
                        'reaction_time_therapist', file_name)

    create_plot_and_set(plots.reaction_time_comparison,
                        (stats['reactions'][1], stats_overall['reactions']['client'],
                         f'{join(path, f"plots/{file_name}/reaction_time_client")}.png'), html,
                        'reaction_time_client', file_name)

    create_plot_and_set(plots.speech_bounds_lengths,
                        (stats['speech_len'][0], stats_overall['speech_len']['therapist'],
                         f'{join(path, f"plots/{file_name}/bounds_len_therapist")}.png'), html,
                        'bounds_len_therapist', file_name)

    create_plot_and_set(plots.speech_bounds_lengths,
                        (stats['speech_len'][1], stats_overall['speech_len']['client'],
                         f'{join(path, f"plots/{file_name}/bounds_len_client")}.png'), html,
                        'bounds_len_client', file_name)

    create_plot_and_set(plots.speed_changes,
                        (stats['transcription'][0], 16, stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}/speed_therapist")}.png'), html,
                        'speed_therapist', file_name)

    create_plot_and_set(plots.speed_changes,
                        (stats['transcription'][1], 16, stats['signal']['len'],
                         f'{join(path, f"plots/{file_name}/speed_client")}.png'), html,
                        'speed_client', file_name)

    create_plot_and_set(plots.hesitations_difference,
                        (stats['hesitations'][0], stats['signal']['len'], stats_overall['hesitations']['therapist'],
                         f'{join(path, f"plots/{file_name}/hesitations_therapist")}.png'), html,
                        'hesitations_therapist', file_name)

    create_plot_and_set(plots.hesitations_difference,
                        (stats['hesitations'][1], stats['signal']['len'], stats_overall['hesitations']['client'],
                         f'{join(path, f"plots/{file_name}/hesitations_client")}.png'), html,
                        'hesitations_client', file_name)

    create_plot_and_set(plots.fills_difference,
                        (stats['fills'][0], stats['signal']['len'], stats_overall['fills']['therapist'],
                         f'{join(path, f"plots/{file_name}/therapist_fills")}.png'), html,
                        'therapist_fills', file_name)

    create_plot_and_set(plots.fills_difference,
                        (stats['fills'][1], stats['signal']['len'], stats_overall['fills']['client'],
                         f'{join(path, f"plots/{file_name}/client_fills")}.png'), html,
                        'client_fills', file_name)

    create_plot_and_set(plots.client_mood,
                        (stats['client_mood'],
                         f'{join(path, f"plots/{file_name}/client_mood")}.png'), html,
                        'client_mood', file_name)


def add_volume_table(html, stats):
    """Add table of most energetic segments with theirs annotations"""
    t_body = html.find(id='volume_text_table').find("tbody")

    loudness = {**dict(((*key, 0), value) for (key, value) in stats['transcription'][0].items() if value['text']),
                **dict(((*key, 1), value) for (key, value) in stats['transcription'][1].items() if value['text'])}
    loud_sorted = sorted(loudness, key=lambda l: loudness[l]['energy'], reverse=True)
    loud_sorted = loud_sorted[0: min(len(loud_sorted), 20)]

    for segment in loud_sorted:
        text = loudness[segment]['text']
        volume = loudness[segment]['energy']
        speaker = segment[2]
        if text is not None:
            new_row = html.new_tag("tr")
            time_start_tag = html.new_tag('td')
            time_start_tag.string = f'{int(segment[0] * params.window_stride / 60)} m ' \
                                    f'{int(segment[0] * params.window_stride % 60)} s'
            new_row.append(time_start_tag)

            time_dur_tag = html.new_tag('td')
            time_dur_tag.string = f'{(segment[1] - segment[0]) * params.window_stride:.2f}'
            new_row.append(time_dur_tag)

            speaker_tag = html.new_tag('td')
            speaker_tag.string = 'Terapeut' if speaker == 0 else 'Klient'
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
                                                                            'speech_ratio']['therapist'])
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


def create_output(stats, stats_overall, texts, template, path, file_name, stats_only):
    """Create single paged html document with overall statistics of session from provided template"""

    with open(join(path, f'stats/{file_name}.pkl'), 'wb') as file:
        pickle.dump(stats, file, pickle.HIGHEST_PROTOCOL)

    if not stats_only:
        html = load_template(template)
        html.find(rel="stylesheet")['href'] = 'style.css'
        add_texts(html, stats, stats_overall, texts, file_name)
        add_attachments(html, file_name)
        add_volume_table(html, stats)
        add_plots(html, stats, stats_overall, path, file_name)
        with open(f'{join(path, file_name)}.html', 'w') as output:
            output.write(str(html))
