"""filename = "hello.mp4"
print(filename[:-4]+"_depth_map"+filename[-4:])"""
"""import time
import sys

def print_progress_bar(iteration, total, length=50, fill='█', prefix='', suffix=''):
    percent = ("{0:.1f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    sys.stdout.write(f'\r{prefix} |{bar}| {percent}% {suffix}')
    sys.stdout.flush()
    if iteration == total:
        sys.stdout.write('\n')
        sys.stdout.flush()

# Example usage
total = 100
for i in range(total + 1):
    time.sleep(0.1)
    print_progress_bar(i, total, prefix='Progress:', suffix='Complete')"""

'''my_string = "this_is_a_file_path.mp4"
print(my_string[my_string.rfind('/')+1:])'''


stuff = [1,2,3,4,5]
new_stuff = stuff[3:4]
print(new_stuff)