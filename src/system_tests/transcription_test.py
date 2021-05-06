import re
from os.path import join
from jiwer import wer
from io_operations.load_files import load_transcription


def transcription_test(ref_path, filename, hyp_path, verbose):
    ref = load_transcription(join(ref_path, filename))
    hyp = load_transcription(join(hyp_path, filename))

    ref_texts = ['', '']
    hyp_texts = ['', '']

    for index, speaker in enumerate(ref):
        for transcription in speaker:
            text = re.sub(r'%[a-zA-Z]*|{[a-zA-Z ]*}|\[[a-zA-Z ]*]|<[a-zA-Z]*|[^a-zA-Z\s]', "",
                          speaker[transcription]['text']).strip()
            ref_texts[index] += f'{text} '

    for index, speaker in enumerate(hyp):
        for transcription in speaker:
            if speaker[transcription]['text'] is not None:
                text = re.sub(r'%[a-zA-Z]*|{[a-zA-Z ]*}|\[[a-zA-Z ]*]|<[a-zA-Z]*|[^a-zA-Z\s]', "",
                              speaker[transcription]['text']).strip()
                hyp_texts[index] += f'{text} '

    error0 = wer(ref_texts[0], hyp_texts[0])
    error1 = wer(ref_texts[1], hyp_texts[1])
    if verbose:
        print(
            f'Word error rate channel 1: {error0 * 100:.3f}%')
        print(
            f'Word error rate channel 2: {error1 * 100:.3f}%')

    return error0, error1
