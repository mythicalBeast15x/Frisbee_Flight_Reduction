import sys
import time


def print_progress_bar(iteration, total, length=50, fill='█', prefix='', suffix='', start_time=-1):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        if start_time != -1:
            end_time = time.time()
            duration = end_time-start_time
            sys.stdout.write(f'\tDuration: {round(duration, 3)} seconds')
        sys.stdout.write('\n')
        sys.stdout.flush()
